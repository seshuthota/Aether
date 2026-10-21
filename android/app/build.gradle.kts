plugins {
    id("com.android.application")
}

android {
    namespace = "dev.aether.chat"
    compileSdk = 36

    defaultConfig {
        applicationId = "dev.aether.chat"
        // Matches the NDK target the engine is built for (android31).
        minSdk = 31
        targetSdk = 36
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += "arm64-v8a" }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    // libaether.so and libaether_jni.so are built by scripts/android-native.sh
    // into src/main/jniLibs/arm64-v8a/.
    packaging {
        jniLibs {
            useLegacyPackaging = false
            // Already stripped by the script.
            keepDebugSymbols += "**/*.so"
        }
    }
}
