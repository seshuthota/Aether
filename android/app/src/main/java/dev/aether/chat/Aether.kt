package dev.aether.chat

import java.nio.ByteBuffer
import java.nio.CharBuffer
import java.nio.charset.CodingErrorAction

/**
 * A loaded model: the engine's C interface (include/aether.h) through the
 * JNI shim in src/main/cpp/aether_jni.c. Use one instance from one thread
 * at a time; the engine pins the thread that opens it to the fastest core,
 * so open and generate on the same thread.
 */
class Aether private constructor(private var handle: Long) : AutoCloseable {
    /** Receives raw UTF-8 bytes; return false to stop generating. */
    fun interface OnText {
        fun onText(bytes: ByteArray): Boolean
    }

    class Stats(val prefillTokens: Int, val prefillMs: Double, val decodeTokens: Int, val decodeMs: Double)

    /** Generates a reply to [prompt]; [chat] wraps it in the ChatML template. */
    fun generate(prompt: String, chat: Boolean, temperature: Float, topP: Float, maxTokens: Int, seed: Long, onText: OnText): Stats {
        check(handle != 0L) { "session is closed" }
        val s = nativeGenerate(handle, prompt.toByteArray(), chat, temperature, topP, maxTokens, seed, onText)
        return Stats(s[0].toInt(), s[1], s[2].toInt(), s[3])
    }

    override fun close() {
        nativeClose(handle)
        handle = 0
    }

    companion object {
        init {
            System.loadLibrary("aether_jni")
        }

        /** Loads a .gguf file; [cores] like "2-7" (one thread per core). */
        fun open(path: String, cores: String, maxSeq: Int) = Aether(nativeOpen(path.toByteArray(), cores.toByteArray(), maxSeq))

        @JvmStatic private external fun nativeOpen(path: ByteArray, cores: ByteArray, maxSeq: Int): Long

        @JvmStatic private external fun nativeGenerate(
            handle: Long, prompt: ByteArray, chat: Boolean, temperature: Float, topP: Float,
            maxTokens: Int, seed: Long, onText: OnText,
        ): DoubleArray

        @JvmStatic private external fun nativeClose(handle: Long)
    }
}

/**
 * Turns pieces of UTF-8 into text. A piece can end in the middle of a
 * character (one emoji can span several tokens); those bytes wait here
 * until the next piece completes them.
 */
class Utf8Stream {
    private val decoder = Charsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPLACE)
        .onUnmappableCharacter(CodingErrorAction.REPLACE)
    private var pending = ByteArray(0)

    fun feed(bytes: ByteArray): String {
        val input = ByteBuffer.wrap(pending + bytes)
        val out = CharBuffer.allocate(input.remaining())
        decoder.decode(input, out, false)
        pending = ByteArray(input.remaining()).also { input.get(it) }
        return out.flip().toString()
    }
}
