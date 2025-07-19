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

//  implementation("org.incendo", "cloud-paper", "2.0.0-beta.10")

//    runtime("ai.djl.pytorch:pytorch-engine:0.33.0")
//    implementation("ai.djl:api:0.28.0")
//    implementation("org.slf4j:slf4j-simple:1.7.36")

//  implementation("org.bytedeco:pytorch:2.7.0-1.5.12-20250515.041143-31")
//  implementation("org.bytedeco:pytorch:2.7.0-1.5.12-20250515.041143-31:windows-x86_64")

//                               pytorch-2.7.1-1.5.12-20250613.191229-12-windows-x86_64.jar
    implementation("org.bytedeco:pytorch:2.7.1-1.5.12-20250613.191229-12")
    implementation("org.bytedeco:pytorch:2.7.1-1.5.12-20250613.191229-12:windows-x86_64")
//    implementation("org.bytedeco:pytorch:2.7.0-1.5.12-20250613.043415-65:windows-x86_64-gpu")

//    implementation("org.bytedeco:openblas:0.3.29-1.5.12-20250319.041331-18")
//    implementation("org.bytedeco:openblas:0.3.29-1.5.12-20250319.041331-18:windows-x86_64")
//                               openblas-0.3.29-1.5.12-20250601.130149-30-windows-x86_64.jar
    implementation("org.bytedeco:openblas:0.3.29-1.5.12-20250601.130149-30:windows-x86_64")


//    implementation("org.bytedeco:mkl:
}

tasks {
    compileJava {
        options.release = 21
    }
    javadoc {
        options.encoding = Charsets.UTF_8.name()
    }

    shadowJar {
        archiveClassifier.set("")
        mergeServiceFiles()

//        fun reloc(pkg: String) = relocate(pkg, "com.tenbitmelon.machinelearningplayer.shadow.$pkg")

//    reloc("org.bytedeco.javacpp")
//    reloc("org.bytedeco.pytorch")
//    reloc("org.bytedeco.openblas")
    }
}

tasks.register<Copy>("copyJarToServer") {
    dependsOn(tasks.shadowJar)
    from(tasks.shadowJar.get().archiveFile)
    into("C:\\Users\\Aidan\\Desktop\\Minecraft Servers\\Paper Server (Machine Learning Player)\\plugins\\")
}