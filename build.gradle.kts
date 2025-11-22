plugins {
    `java-library`
    id("io.papermc.paperweight.userdev") version "2.0.0-beta.17"
    id("xyz.jpenilla.run-paper") version "2.3.1"

//    id("org.bytedeco.gradle-javacpp-platform") version "1.5.12"
    id("org.bytedeco.gradle-javacpp-platform") version "1.5.10"

    id("com.gradleup.shadow") version "8.3.6"
}

group = "com.tenbitmelon.machinelearningplayer"
version = "1.0.0-SNAPSHOT"

java {
    toolchain.languageVersion = JavaLanguageVersion.of(21)
}

ext.set("javacppPlatform", "windows-x86_64")

repositories {
    mavenCentral()
    maven {
        name = "sonatype-snapshots"
        url = uri("https://oss.sonatype.org/content/repositories/snapshots")
    }
    maven {
        name = "sonatype"
        url = uri("https://oss.sonatype.org/content/groups/public/")
    }
}

dependencies {
    paperweight.paperDevBundle("1.21.5-R0.1-SNAPSHOT")

////  implementation("org.incendo", "cloud-paper", "2.0.0-beta.10")
//
////    runtime("ai.djl.pytorch:pytorch-engine:0.33.0")
////    implementation("ai.djl:api:0.28.0")
////    implementation("org.slf4j:slf4j-simple:1.7.36")
//
////  implementation("org.bytedeco:pytorch:2.7.0-1.5.12-20250515.041143-31")
////  implementation("org.bytedeco:pytorch:2.7.0-1.5.12-20250515.041143-31:windows-x86_64")
//
////                               pytorch-2.7.1-1.5.12-20250613.191229-12-windows-x86_64.jar
//    implementation("org.bytedeco:pytorch:2.7.1-1.5.12-20250613.191229-12")
//    implementation("org.bytedeco:pytorch:2.7.1-1.5.12-20250613.191229-12:windows-x86_64")
//
//    // pytorch-2.7.1-1.5.12-20250613.193524-13-windows-x86_64-gpu.jar
//    implementation("org.bytedeco:pytorch:2.7.1-1.5.12-20250613.193524-13:windows-x86_64-gpu")
//
//    // cuda-12.9-9.10-1.5.12-20250612.145546-3-windows-x86_64-redist.jar
//    implementation("org.bytedeco:cuda:12.9-9.10-1.5.12-20250612.145546-3:windows-x86_64-redist")
//
////    implementation("org.bytedeco:openblas:0.3.29-1.5.12-20250319.041331-18")
////    implementation("org.bytedeco:openblas:0.3.29-1.5.12-20250319.041331-18:windows-x86_64")
////                               openblas-0.3.29-1.5.12-20250601.130149-30-windows-x86_64.jar
//    implementation("org.bytedeco:openblas:0.3.29-1.5.12-20250601.130149-30:windows-x86_64")


    // javacpp-1.5.12-20250613.133933-85-windows-x86_64.jar
    implementation("org.bytedeco:javacpp:1.5.12-20250613.133933-85:windows-x86_64")

    implementation("org.bytedeco:pytorch:2.7.1-1.5.12-20250613.193524-13")
    implementation("org.bytedeco:pytorch:2.7.1-1.5.12-20250613.193524-13:windows-x86_64-gpu")

//    implementation("org.bytedeco:cuda:12.9-9.10-1.5.12-20250612.145546-3:windows-x86_64-redist")
    // cuda-12.9-9.10-1.5.12-20250612.145546-3.jar
    implementation("org.bytedeco:cuda:12.9-9.10-1.5.12-20250612.145546-3")
    // cuda-12.9-9.10-1.5.12-20250612.145546-3-windows-x86_64.jar
    implementation("org.bytedeco:cuda:12.9-9.10-1.5.12-20250612.145546-3:windows-x86_64")

    implementation("org.bytedeco:openblas:0.3.29-1.5.12-20250601.130149-30:windows-x86_64")

    // mkl-2025.2-1.5.12-windows-x86_64.jar
    implementation("org.bytedeco:mkl:2025.2-1.5.12:windows-x86_64")

//    implementation("org.bytedeco:pytorch-platform-gpu:2.7.1-1.5.12")
//
//    implementation("org.bytedeco:cuda-platform:12.9-9.10-1.5.12")
//    implementation("org.bytedeco:cuda:12.9-9.10-1.5.12:windows-x86_64")


//    //                        pytorch-2.7.1-1.5.12-windows-x86_64.jar
//    implementation("org.bytedeco:pytorch:2.7.1-1.5.12")
//    //                        pytorch-2.7.1-1.5.12-windows-x86_64-gpu.jar
//    implementation("org.bytedeco:pytorch:2.7.1-1.5.12:windows-x86_64-gpu")
//    //                        cuda-platform-redist-12.6-9.5-1.5.11.jar
//    //                        cuda-12.9-9.10-1.5.12-windows-x86_64.jar
//    implementation("org.bytedeco:cuda:12.9-9.10-1.5.12:windows-x86_64")
//    //                        openblas-0.3.30-1.5.12-windows-x86_64.jar
//    implementation("org.bytedeco:openblas:0.3.29-1.5.12:windows-x86_64")

}

tasks {
    compileJava {
        options.release = 21
    }
    javadoc {
        options.encoding = Charsets.UTF_8.name()
    }

    jar {
        manifest {
            attributes(
                "Class-Path" to configurations.runtimeClasspath.get().joinToString(" ") { "lib/${it.name}" },
            )
        }
    }

    shadowJar {
        archiveClassifier.set("")
        mergeServiceFiles()

        dependencies {
            exclude(dependency("org.bytedeco:.*"))
        }
    }
}

tasks.register<Copy>("copyDependencies") {
    from(configurations.runtimeClasspath) {
//        include("*.jar")
        include("**/*.jar")
    }
    into("C:\\Users\\Aidan\\Desktop\\Minecraft Servers\\Paper Server (Machine Learning Player)\\plugins\\lib\\")
}

tasks.register<Copy>("copyJarToServer") {
    dependsOn(tasks.shadowJar, "copyDependencies")
    from(tasks.shadowJar.get().archiveFile)
    into("C:\\Users\\Aidan\\Desktop\\Minecraft Servers\\Paper Server (Machine Learning Player)\\plugins\\")
}