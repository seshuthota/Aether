//! A background sampler of clocks, temperatures and battery power.
//!
//! A benchmark on a phone measures two things at once: the code, and the
//! state of the SoC while it ran. Clocks drop as the chip heats up, and the
//! battery current says how much energy the run cost. [`Monitor::start`]
//! spawns a thread that reads those `/sys` files every `period` until
//! [`Monitor::stop`] is called; the [`Report`] then gives the run's energy in
//! joules, average power, and how hot and how throttled the cores got.
//!
//! Energy is power integrated over time. Power is `|current| × voltage`; with
//! samples at times `t0, t1, ..` the integral is approximated by trapezoids:
//! `E ≈ Σ (P_i + P_{i+1}) / 2 × (t_{i+1} − t_i)`.
//!
//! Only a run on battery means anything for energy. While the phone is
//! plugged in, `current_now` is the charging current. [`Report::summary`]
//! says so when the battery was not discharging.

use crate::sys::affinity;
use crate::sys::sysfs::{self, Battery, ThermalZone};
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// One reading of every source. `None` means that file could not be read at
/// that moment (it is kept, not dropped, so columns stay aligned).
#[derive(Debug, Clone)]
pub struct Sample {
    /// Seconds since the monitor started.
    pub t_s: f64,
    /// Current clock per core, in the order of [`Report::cpus`].
    pub freq_khz: Vec<Option<u64>>,
    /// Temperature per zone, in the order of [`Report::zones`].
    pub temp_millic: Vec<Option<i64>>,
    pub current_ua: Option<i64>,
    pub voltage_uv: Option<i64>,
    /// Remaining battery charge, µAh (falls while discharging).
    pub counter_uah: Option<i64>,
}

impl Sample {
    /// Battery power in watts: |current| × voltage (µA × µV = 1e-12 W).
    pub fn power_w(&self) -> Option<f64> {
        Some((self.current_ua? as f64).abs() * self.voltage_uv? as f64 * 1e-12)
    }
}

/// Everything a monitored run recorded.
#[derive(Debug, Clone)]
pub struct Report {
    pub cpus: Vec<usize>,
    pub zones: Vec<ThermalZone>,
    /// Battery status when the monitor started (`Discharging`, `Charging`, ..).
    pub battery_status: Option<String>,
    pub samples: Vec<Sample>,
}

/// A running sampler. Dropping it without calling [`Monitor::stop`] stops
/// the thread too, but loses the report.
pub struct Monitor {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<Vec<Sample>>>,
    cpus: Vec<usize>,
    zones: Vec<ThermalZone>,
    battery_status: Option<String>,
}

/// Zones worth sampling: CPU sensors if the device names any (the phone has
/// dozens of zones, most of them irrelevant), otherwise all readable zones.
fn interesting_zones() -> Vec<ThermalZone> {
    let all = sysfs::thermal_zones();
    let cpu: Vec<ThermalZone> = all.iter().filter(|z| z.kind.to_lowercase().contains("cpu")).cloned().collect();
    if cpu.is_empty() { all } else { cpu }
}

fn read_sample(start: Instant, cpus: &[usize], zones: &[ThermalZone], battery: Option<&Battery>) -> Sample {
    Sample {
        t_s: start.elapsed().as_secs_f64(),
        freq_khz: cpus.iter().map(|&c| sysfs::cpu_cur_freq_khz(c).ok()).collect(),
        temp_millic: zones.iter().map(|z| z.temp_millic().ok()).collect(),
        current_ua: battery.and_then(|b| b.current_ua().ok()),
        voltage_uv: battery.and_then(|b| b.voltage_uv().ok()),
        counter_uah: battery.and_then(|b| b.counter_uah().ok()),
    }
}

impl Monitor {
    /// Starts sampling every `period`. If `pin` is given, the sampler thread
    /// runs only on those cores, so it does not compete with a benchmark
    /// pinned elsewhere.
    pub fn start(period: Duration, pin: Option<&[usize]>) -> io::Result<Monitor> {
        let cpus = affinity::online_cpus()?;
        let zones = interesting_zones();
        let battery = Battery::find();
        let battery_status = battery.as_ref().and_then(|b| b.status().ok());
        let stop = Arc::new(AtomicBool::new(false));
        let pin = pin.map(|p| p.to_vec());

        // The thread gets its own copies; the originals stay for the report.
        let (t_stop, t_cpus, t_zones) = (stop.clone(), cpus.clone(), zones.clone());
        let handle = thread::Builder::new().name("aether-monitor".into()).spawn(move || {
            if let Some(cores) = pin {
                // A failed pin only makes the sampler noisier; keep sampling.
                let _ = affinity::pin_current_thread(&cores);
            }
            let start = Instant::now();
            let mut samples = Vec::new();
            let mut next = start;
            // Android's battery service costs a subprocess per read and only
            // refreshes every ~minute: poll it every 5 s, not every sample.
            let slow_period = Duration::from_secs(5);
            let mut last_battery: Option<Instant> = None;
            loop {
                let due = battery.as_ref().is_some_and(|b| b.is_fast() || last_battery.is_none_or(|t| t.elapsed() >= slow_period));
                if due {
                    last_battery = Some(Instant::now());
                }
                samples.push(read_sample(start, &t_cpus, &t_zones, if due { battery.as_ref() } else { None }));
                if t_stop.load(Ordering::Acquire) {
                    return samples; // the last sample is taken after `stop`
                }
                // Sleep until the next tick on a fixed grid (no drift from
                // the time spent reading). `park_timeout` rather than
                // `sleep` so `stop` can wake the thread immediately.
                next += period;
                let now = Instant::now();
                if next > now {
                    thread::park_timeout(next - now);
                } else {
                    next = now; // fell behind: restart the grid from now
                }
            }
        })?;
        Ok(Monitor { stop, handle: Some(handle), cpus, zones, battery_status })
    }

    /// Stops the sampler (after one final sample) and returns the report.
    pub fn stop(mut self) -> Report {
        let samples = self.finish();
        Report {
            cpus: std::mem::take(&mut self.cpus),
            zones: std::mem::take(&mut self.zones),
            battery_status: self.battery_status.take(),
            samples,
        }
    }

    fn finish(&mut self) -> Vec<Sample> {
        let Some(h) = self.handle.take() else { return Vec::new() };
        self.stop.store(true, Ordering::Release);
        h.thread().unpark();
        h.join().expect("monitor thread panicked")
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        self.finish();
    }
}

impl Report {
    /// Seconds between the first and last sample.
    pub fn duration_s(&self) -> f64 {
        match (self.samples.first(), self.samples.last()) {
            (Some(a), Some(b)) => b.t_s - a.t_s,
            _ => 0.0,
        }
    }

    /// Energy drawn from the battery over the run, in joules (trapezoid
    /// rule). `None` if fewer than two samples have a power reading.
    pub fn energy_j(&self) -> Option<f64> {
        let pts: Vec<(f64, f64)> = self.samples.iter().filter_map(|s| Some((s.t_s, s.power_w()?))).collect();
        if pts.len() < 2 {
            return None;
        }
        Some(pts.windows(2).map(|w| (w[0].1 + w[1].1) / 2.0 * (w[1].0 - w[0].0)).sum())
    }

    /// Average battery power over the run, in watts.
    pub fn avg_power_w(&self) -> Option<f64> {
        let d = self.duration_s();
        if d > 0.0 { Some(self.energy_j()? / d) } else { None }
    }

    /// Energy from the battery's charge counter: `(joules, window_s)`.
    ///
    /// Where the current reading is slow or missing (Android's battery
    /// service refreshes about once a minute) the counter is the better
    /// source. It changes in steps, at the moments the service refreshes, so
    /// the charge drawn is known exactly between the first and the last step
    /// seen during the run: `ΔQ = c(first step) − c(last step)` over that
    /// window. Energy = ΔQ × average voltage (1 µAh = 3.6 mC). `None` until
    /// the counter has stepped at least twice.
    pub fn counter_energy_j(&self) -> Option<(f64, f64)> {
        let pts: Vec<(f64, i64)> = self.samples.iter().filter_map(|s| Some((s.t_s, s.counter_uah?))).collect();
        let steps: Vec<(f64, i64)> = pts.windows(2).filter(|w| w[1].1 != w[0].1).map(|w| w[1]).collect();
        let (&(t_a, c_a), &(t_b, c_b)) = (steps.first()?, steps.last()?);
        if steps.len() < 2 || t_b <= t_a {
            return None;
        }
        let volts: Vec<f64> = self.samples.iter().filter_map(|s| Some(s.voltage_uv? as f64 * 1e-6)).collect();
        if volts.is_empty() {
            return None;
        }
        let v_avg = volts.iter().sum::<f64>() / volts.len() as f64;
        Some(((c_a - c_b) as f64 * 3.6e-3 * v_avg, t_b - t_a))
    }

    /// Best available average power, in watts, and which source gave it:
    /// the charge counter when it stepped at least twice, else the
    /// integrated current × voltage.
    pub fn best_avg_power_w(&self) -> Option<(f64, &'static str)> {
        if let Some((e, window)) = self.counter_energy_j() {
            return Some((e / window, "charge counter"));
        }
        self.avg_power_w().map(|p| (p, "current x voltage"))
    }

    /// Hottest reading of any sampled zone, in °C.
    pub fn max_temp_c(&self) -> Option<f64> {
        self.samples.iter().flat_map(|s| s.temp_millic.iter().flatten()).max().map(|&m| m as f64 / 1000.0)
    }

    /// Lowest clock each core was seen at, in kHz (shows throttling).
    pub fn min_freq_khz(&self) -> Vec<(usize, Option<u64>)> {
        self.cpus
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, self.samples.iter().filter_map(|s| s.freq_khz[i]).min()))
            .collect()
    }

    /// One-paragraph summary for the end of a benchmark's output.
    pub fn summary(&self) -> String {
        let mut s = format!("monitor: {} samples over {:.2} s", self.samples.len(), self.duration_s());
        if let Some(t) = self.max_temp_c() {
            s += &format!(", max temp {t:.1} °C");
        }
        match (self.energy_j(), self.avg_power_w()) {
            (Some(e), Some(p)) => s += &format!(", energy {e:.2} J, avg power {p:.2} W"),
            _ => s += ", no battery power readings",
        }
        if let Some((e, window)) = self.counter_energy_j() {
            s += &format!(", charge counter: {e:.1} J over a {window:.0} s window = {:.2} W", e / window);
        }
        let status = self.battery_status.as_deref().unwrap_or("unknown");
        if status != "Discharging" {
            s += &format!(" (battery {status}: energy is NOT the run's consumption)");
        }
        s
    }

    /// All samples as CSV: `t_s, cpuN_khz.., <zone>_mc.., current_ua,
    /// voltage_uv, power_w`. Unreadable values are empty cells.
    pub fn write_csv(&self, mut w: impl Write) -> io::Result<()> {
        let mut header = vec!["t_s".to_string()];
        header.extend(self.cpus.iter().map(|c| format!("cpu{c}_khz")));
        header.extend(self.zones.iter().map(|z| format!("tz{}_{}_mc", z.index, z.kind)));
        header.extend(["current_ua", "voltage_uv", "counter_uah", "power_w"].map(String::from));
        writeln!(w, "{}", header.join(","))?;
        let cell = |v: Option<String>| v.unwrap_or_default();
        for s in &self.samples {
            let mut row = vec![format!("{:.3}", s.t_s)];
            row.extend(s.freq_khz.iter().map(|v| cell(v.map(|x| x.to_string()))));
            row.extend(s.temp_millic.iter().map(|v| cell(v.map(|x| x.to_string()))));
            row.push(cell(s.current_ua.map(|x| x.to_string())));
            row.push(cell(s.voltage_uv.map(|x| x.to_string())));
            row.push(cell(s.counter_uah.map(|x| x.to_string())));
            row.push(cell(s.power_w().map(|x| format!("{x:.4}"))));
            writeln!(w, "{}", row.join(","))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(t_s: f64, current_ua: i64, voltage_uv: i64) -> Sample {
        Sample { t_s, freq_khz: vec![], temp_millic: vec![], current_ua: Some(current_ua), voltage_uv: Some(voltage_uv), counter_uah: None }
    }

    fn report(samples: Vec<Sample>) -> Report {
        Report { cpus: vec![], zones: vec![], battery_status: Some("Discharging".into()), samples }
    }

    /// 1 A at 4 V is 4 W; for 2 s that is 8 J, whichever sign the driver
    /// uses for discharge current.
    #[test]
    fn constant_power_integrates_to_power_times_time() {
        let r = report(vec![sample(0.0, -1_000_000, 4_000_000), sample(1.0, -1_000_000, 4_000_000), sample(2.0, 1_000_000, 4_000_000)]);
        assert!((r.energy_j().unwrap() - 8.0).abs() < 1e-9);
        assert!((r.avg_power_w().unwrap() - 4.0).abs() < 1e-9);
    }

    /// Power rising linearly from 0 W to 2 W over 1 s: the trapezoid is exact,
    /// 1 J.
    #[test]
    fn trapezoid_is_exact_for_a_ramp() {
        let r = report(vec![sample(0.0, 0, 4_000_000), sample(1.0, 500_000, 4_000_000)]);
        assert!((r.energy_j().unwrap() - 1.0).abs() < 1e-9);
    }

    /// Counter steps at t = 10 s and t = 70 s: 10 000 µAh in 60 s is 600 mA;
    /// at 4 V that is 2.4 W, 144 J. Samples before the first step and after
    /// the last are outside the known window and must not count.
    #[test]
    fn counter_energy_uses_the_window_between_steps() {
        let mut samples = Vec::new();
        for t in 0..90 {
            let c = if t < 10 { 1_000_000 } else if t < 70 { 990_000 } else { 980_000 };
            samples.push(Sample { counter_uah: Some(c), ..sample(t as f64, 0, 4_000_000) });
        }
        let r = report(samples);
        let (e, window) = r.counter_energy_j().unwrap();
        assert!((window - 60.0).abs() < 1e-9 && (e - 144.0).abs() < 1e-9, "{e} J over {window} s");
        assert_eq!(r.best_avg_power_w().unwrap().1, "charge counter");
    }

    /// The real thread: it samples on its grid, stops promptly, and the CSV
    /// has one header plus one line per sample.
    #[test]
    fn samples_until_stopped() {
        let m = Monitor::start(Duration::from_millis(20), None).unwrap();
        // On Android the first sample reads the battery service (three
        // subprocesses): over 150 ms while the other tests load every core.
        // Later samples skip it until the slow period. 1 s once gave only 1
        // sample (the 112 tests with the DSP's on the phone), so 3 s.
        thread::sleep(Duration::from_secs(3));
        let r = m.stop();
        assert!(r.samples.len() >= 5, "only {} samples", r.samples.len());
        // Stopping finishes the sample in progress (the first one reads the
        // battery: up to ~1 s of subprocesses under load) but must not wait
        // for the next tick. With a 10 s period the two are far apart.
        let t = Instant::now();
        Monitor::start(Duration::from_secs(10), None).unwrap().stop();
        assert!(t.elapsed() < Duration::from_secs(5), "stop waited {:?}: it must not wait for the next tick", t.elapsed());
        assert!(r.samples.windows(2).all(|w| w[0].t_s <= w[1].t_s));
        let mut csv = Vec::new();
        r.write_csv(&mut csv).unwrap();
        assert_eq!(String::from_utf8(csv).unwrap().lines().count(), r.samples.len() + 1);
        println!("{}", r.summary());
    }
}
