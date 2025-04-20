#!/usr/bin/env kotlinc -script
@file:DependsOn("com.microsoft.onnxruntime:onnxruntime:1.15.1")

import java.io.File
import java.nio.charset.Charset
import ai.onnxruntime.*

/**
 * Simple interface for a line‑level classifier.
 */
interface ErrorClassifier {
    fun predict(lines: List<String>): List<Boolean>
}

/**
 * Loads an ONNX‑exported sklearn pipeline (including text vectorizer + classifier)
 * and applies it to raw log lines.
 */
class OnnxErrorClassifier(modelPath: String) : ErrorClassifier {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val outputName: String

    init {
        val modelFile = File(modelPath)
        require(modelFile.exists()) { "ONNX model not found: $modelPath" }
        session = env.createSession(modelFile.absolutePath, OrtSession.SessionOptions())
        inputName = session.inputNames.first()
        outputName = session.outputNames.first()
    }

    override fun predict(lines: List<String>): List<Boolean> {
        // create a 1-D string tensor of shape [N]
        val tensor = OnnxTensor.createTensor(env, lines.toTypedArray(), longArrayOf(lines.size.toLong()))
        session.use { sess ->
            tensor.use { input ->
                val result = sess.run(mapOf(inputName to input))
                val out = result[outputName] as OnnxTensor
                // assume output is a tensor of shape [N] with ints 0/1
                val raw = out.value
                return when (raw) {
                    is LongArray -> raw.map { it.toInt() == 1 }
                    is Array<LongArray> -> raw.map { it.first().toInt() == 1 }
                    is FloatArray -> raw.map { it > 0.5f }
                    is Array<FloatArray> -> raw.map { it.first() > 0.5f }
                    else -> error("Unexpected output tensor type: ${raw::class}")
                }
            }
        }
    }
}

/**
 * Filters a log file for lines matching known error patterns
 * or classified as errors by the ONNX model.
 */
class ErrorLogFilter(modelPath: String) {
    private val classifier = OnnxErrorClassifier(modelPath)
    private val regexes = listOf(
        // Compilation errors
        "error[:\\s]", "exception in", "failed with exit code", "compilation failed", "build failed",
        // Java/JVM
        "nullpointerexception", "classnotfoundexception", "outofmemoryerror", "stackoverflowerror",
        // JS/Node
        "cannot find module", "unexpected token", "is not defined", "is not a function",
        // Python
        "importerror", "indentationerror", "syntaxerror", "nameerror",
        // Build tools
        "could not resolve", "dependency not found", "failed to resolve",
        // Docker
        "image not found", "container exited"
    ).map { it.toRegex(RegexOption.IGNORE_CASE) }

    private fun loadLines(path: String): List<String> {
        val file = File(path)
        require(file.exists()) { "Log file not found: $path" }
        return file.readLines(Charset.forName("UTF-8"))
    }

    private fun isError(line: String): Boolean {
        if (regexes.any { it.containsMatchIn(line) }) return true
        return classifier.predict(listOf(line)).first()
    }

    fun printErrorsFromFile(logPath: String) {
        val lines = loadLines(logPath)
        lines.filter { isError(it) }
             .forEach(::println)
    }
}

fun main(args: Array<String>) {
    var modelPath = "error_classifier.onnx"
    var logPath   = "input_errors.txt"

    for (i in args.indices) {
        when (args[i]) {
            "-m", "--model" -> if (i + 1 < args.size) modelPath = args[i + 1]
            "-l", "--log"   -> if (i + 1 < args.size) logPath = args[i + 1]
        }
    }

    ErrorLogFilter(modelPath).printErrorsFromFile(logPath)
}
