//! Readers for the `/sys` files that describe clocks, heat and the battery.
//!
//! Every file under `/sys` is a small text file the kernel generates when it
//! is read: `cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq` prints
//! one number. Reading one from Rust is: read the text, trim the newline,
//! parse, and attach the unit the kernel documents. Names below carry the
//! unit (`_khz`, `_millic`, `_ua`, `_uv`) so a number never loses it.
//!
//! Files can be missing (a laptop has no `scaling_cur_freq` for some
//! drivers), unreadable (Android hides some thermal zones from the shell
//! user) or temporarily invalid (some zones return EAGAIN while a sensor is
//! off). All of those come back as `io::Error` naming the file, never as a
//! silent zero.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// The file's contents with surrounding whitespace (the trailing newline)
/// removed.
pub fn read_trimmed(path: impl AsRef<Path>) -> io::Result<String> {
    let path = path.as_ref();
    fs::read_to_string(path)
        .map(|s| s.trim().to_string())
        .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))
}

/// The file's contents parsed as a number.
pub fn read_num<T: FromStr>(path: impl AsRef<Path>) -> io::Result<T> {
    let path = path.as_ref();
    let s = read_trimmed(path)?;
    s.parse()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, format!("{}: not a number: {s:?}", path.display())))
}

fn cpufreq(cpu: usize, file: &str) -> PathBuf {
    PathBuf::from(format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/{file}"))
}

/// The clock the frequency governor has currently set for `cpu`, in kHz.
/// Cores in one cluster share a clock, so they report the same value.
pub fn cpu_cur_freq_khz(cpu: usize) -> io::Result<u64> {
    read_num(cpufreq(cpu, "scaling_cur_freq"))
}

/// The highest clock `cpu` can run at, in kHz (hardware limit, not the
/// current thermal cap).
pub fn cpu_max_freq_khz(cpu: usize) -> io::Result<u64> {
    read_num(cpufreq(cpu, "cpuinfo_max_freq"))
}

/// The current upper clock limit for `cpu`, in kHz. Thermal throttling shows
/// up here: it drops below `cpu_max_freq_khz` as the SoC heats up.
pub fn cpu_scaling_max_freq_khz(cpu: usize) -> io::Result<u64> {
    read_num(cpufreq(cpu, "scaling_max_freq"))
}

/// One temperature sensor exposed under `/sys/class/thermal/thermal_zoneN`.
#[derive(Debug, Clone)]
pub struct ThermalZone {
    /// The `N` in `thermal_zoneN`.
    pub index: usize,
    /// The sensor's name from its `type` file, e.g. `cpu-1-0-0` or `battery`.
    pub kind: String,
}

impl ThermalZone {
    /// Current temperature in milli-degrees Celsius (`45200` = 45.2 °C).
    pub fn temp_millic(&self) -> io::Result<i64> {
        read_num(format!("/sys/class/thermal/thermal_zone{}/temp", self.index))
    }
}

/// Every thermal zone whose `type` and `temp` can both be read right now,
/// sorted by index. Zones the shell user may not read are skipped.
pub fn thermal_zones() -> Vec<ThermalZone> {
    let Ok(dir) = fs::read_dir("/sys/class/thermal") else {
        return Vec::new();
    };
    let mut zones: Vec<ThermalZone> = dir
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str()?.strip_prefix("thermal_zone")?.parse().ok())
        .filter_map(|index: usize| {
            let kind = read_trimmed(format!("/sys/class/thermal/thermal_zone{index}/type")).ok()?;
            let zone = ThermalZone { index, kind };
            zone.temp_millic().ok()?;
            Some(zone)
        })
        .collect();
    zones.sort_by_key(|z| z.index);
    zones
}

/// Where battery readings come from.
///
/// - `Sysfs`: the entry of `/sys/class/power_supply` whose `type` is
///   `Battery` (`BAT0` on a laptop, `battery` on most phones). Fresh values,
///   cheap to read.
/// - `AndroidService`: Android's battery service, through the `cmd battery`
///   and `dumpsys battery` commands. Used when SELinux denies the shell user
///   the sysfs files, as on the iQOO 13. Each read starts a subprocess
///   (~14 ms), and on that phone the service refreshes its values only every
///   45–85 s (measured, `results/phase-1/raw/battery-counter-cadence.csv`).
#[derive(Debug, Clone)]
pub enum Battery {
    Sysfs(PathBuf),
    AndroidService,
}

/// Runs a shell command and returns its trimmed stdout.
fn command_output(program: &str, args: &[&str]) -> io::Result<String> {
    let out = std::process::Command::new(program).args(args).output()?;
    if !out.status.success() {
        return Err(io::Error::other(format!("{program} {args:?} failed: {}", out.status)));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The number after `key` in `dumpsys battery` output (`  voltage: 4258`).
fn dumpsys_battery_field(key: &str) -> io::Result<i64> {
    let text = command_output("dumpsys", &["battery"])?;
    text.lines()
        .find_map(|l| l.trim().strip_prefix(key)?.trim().parse().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("no `{key}` in dumpsys battery")))
}

fn cmd_battery_get(what: &str) -> io::Result<i64> {
    let s = command_output("cmd", &["battery", "get", what])?;
    s.parse().map_err(|_| io::Error::new(io::ErrorKind::InvalidData, format!("cmd battery get {what}: {s:?}")))
}

impl Battery {
    /// A readable sysfs battery if there is one; otherwise Android's battery
    /// service if this is Android; otherwise `None`.
    pub fn find() -> Option<Battery> {
        let dirs = fs::read_dir("/sys/class/power_supply").ok().map(|d| {
            let mut v: Vec<PathBuf> = d.filter_map(|e| Some(e.ok()?.path())).collect();
            v.sort();
            v
        });
        let sysfs = dirs.into_iter().flatten().find(|d| read_trimmed(d.join("type")).is_ok_and(|t| t == "Battery"));
        match sysfs {
            Some(dir) => Some(Battery::Sysfs(dir)),
            None if Path::new("/system/bin/cmd").exists() => Some(Battery::AndroidService),
            None => None,
        }
    }

    /// True when every read is cheap and fresh (sysfs); false when reads cost
    /// a subprocess and values lag (Android service).
    pub fn is_fast(&self) -> bool {
        matches!(self, Battery::Sysfs(_))
    }

    /// Current flowing out of (or into) the battery, in microamps, as the
    /// driver reports it. The sign convention is the driver's choice (the
    /// iQOO 13 reports discharge as negative), and while charging the value is
    /// the charge current. Use [`Battery::status`] to know which situation a
    /// reading came from.
    pub fn current_ua(&self) -> io::Result<i64> {
        match self {
            Battery::Sysfs(dir) => read_num(dir.join("current_now")),
            // The iQOO 13's service reports mA here (−475 under an all-core
            // load, while the charge counter fell at 477 mA): convert.
            Battery::AndroidService => cmd_battery_get("current_now").map(|ma| ma * 1000),
        }
    }

    /// Battery terminal voltage, in microvolts (`4123000` = 4.123 V).
    pub fn voltage_uv(&self) -> io::Result<i64> {
        match self {
            Battery::Sysfs(dir) => read_num(dir.join("voltage_now")),
            Battery::AndroidService => dumpsys_battery_field("voltage:").map(|mv| mv * 1000),
        }
    }

    /// Remaining charge, in microamp-hours ("charge counter"). Falls as the
    /// battery discharges; the difference over a run is the charge it drew.
    pub fn counter_uah(&self) -> io::Result<i64> {
        match self {
            Battery::Sysfs(dir) => read_num(dir.join("charge_counter")),
            Battery::AndroidService => cmd_battery_get("counter"),
        }
    }

    /// `Charging`, `Discharging`, `Full` or `Not charging`. Energy readings
    /// mean something only while `Discharging`.
    pub fn status(&self) -> io::Result<String> {
        match self {
            Battery::Sysfs(dir) => read_trimmed(dir.join("status")),
            // BatteryManager.BATTERY_STATUS_* codes.
            Battery::AndroidService => dumpsys_battery_field("status:").map(|code| {
                match code {
                    2 => "Charging",
                    3 => "Discharging",
                    4 => "Not charging",
                    5 => "Full",
                    _ => "Unknown",
                }
                .to_string()
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_names_the_path() {
        let e = read_num::<u64>("/sys/nonexistent/aether").unwrap_err();
        assert!(e.to_string().contains("/sys/nonexistent/aether"));
    }

    /// Prints what this machine exposes. On the phone this is the inventory
    /// the sampler builds on; on the host it shows the same code working.
    #[test]
    fn inventory() {
        for cpu in crate::sys::affinity::online_cpus().unwrap() {
            println!(
                "cpu{cpu}: cur {:?} kHz, cap {:?} kHz, max {:?} kHz",
                cpu_cur_freq_khz(cpu).ok(),
                cpu_scaling_max_freq_khz(cpu).ok(),
                cpu_max_freq_khz(cpu).ok()
            );
        }
        for z in thermal_zones() {
            println!("thermal_zone{} {:<24} {} m°C", z.index, z.kind, z.temp_millic().unwrap_or(-1));
        }
        match Battery::find() {
            Some(b) => println!(
                "battery {b:?}: status {:?}, current {:?} uA, voltage {:?} uV, counter {:?} uAh",
                b.status().ok(),
                b.current_ua().ok(),
                b.voltage_uv().ok(),
                b.counter_uah().ok()
            ),
            None => println!("no battery"),
        }
    }

    /// Every core reports a maximum frequency; on the phone the two
    /// clusters must show up as two distinct values.
    #[test]
    fn every_online_core_has_a_max_frequency() {
        for cpu in crate::sys::affinity::online_cpus().unwrap() {
            assert!(cpu_max_freq_khz(cpu).unwrap() > 0);
        }
    }
}
