package dev.aether.chat

import android.app.Activity
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * One screen: pick a model, type a prompt, watch the reply stream in with
 * prefill and decode speeds underneath.
 *
 * Models are .gguf files in the app's own storage: files/ (internal) or
 * Android/data/dev.aether.chat/files/ (shared storage); see android/README.md.
 */
class MainActivity : Activity() {
    // Every engine call runs on this one thread (see Aether).
    private val engine = Executors.newSingleThreadExecutor()
    private var session: Aether? = null
    private var loaded = "" // "path|cores" of the open session
    private val stop = AtomicBoolean(false)
    private var running = false

    private lateinit var models: Spinner
    private lateinit var cores: EditText
    private lateinit var chat: CheckBox
    private lateinit var lookup: CheckBox
    private lateinit var backend: Spinner
    private lateinit var prompt: EditText
    private lateinit var go: Button
    private lateinit var output: TextView
    private lateinit var status: TextView
    private lateinit var scroll: ScrollView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val pad = (12 * resources.displayMetrics.density).toInt()
        models = Spinner(this)
        cores = EditText(this).apply {
            setText("2-7")
            hint = "cores"
            inputType = InputType.TYPE_CLASS_TEXT
            setEms(4)
        }
        chat = CheckBox(this).apply {
            text = "chat"
            isChecked = true
        }
        // Prompt lookup decoding: pays when the reply repeats the prompt.
        lookup = CheckBox(this).apply { text = "lookup" }
        // Where the model runs. GPU: decode on the GPU, the prompt on the
        // CPU cores. DSP: everything on the Hexagon DSP (a --pure Q4_0 or a
        // Q4_K_M model; cores unused).
        backend = Spinner(this).apply {
            adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, BACKENDS)
        }
        prompt = EditText(this).apply {
            hint = "Ask something"
            setText("Why is the sky blue?")
        }
        go = Button(this).apply {
            text = "Generate"
            setOnClickListener { if (running) stop.set(true) else start() }
        }
        output = TextView(this).apply {
            setTextIsSelectable(true)
            textSize = 16f
        }
        status = TextView(this).apply { textSize = 12f }
        scroll = ScrollView(this).apply { addView(output) }

        val settings = LinearLayout(this).apply {
            gravity = Gravity.CENTER_VERTICAL
            addView(models, LinearLayout.LayoutParams(0, -2, 1f))
            addView(cores)
            addView(chat)
            addView(lookup)
            addView(backend)
        }
        val ask = LinearLayout(this).apply {
            gravity = Gravity.CENTER_VERTICAL
            addView(prompt, LinearLayout.LayoutParams(0, -2, 1f))
            addView(go)
        }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(settings)
            addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
            addView(status)
            addView(ask)
        }
        // Content draws behind the system bars and keyboard; pad it clear.
        root.setOnApplyWindowInsetsListener { v, insets ->
            val i = insets.getInsets(WindowInsets.Type.systemBars() or WindowInsets.Type.ime())
            v.setPadding(i.left + pad, i.top + pad, i.right + pad, i.bottom + pad)
            WindowInsets.CONSUMED
        }
        setContentView(root)
        listModels()
    }

    override fun onResume() {
        super.onResume()
        if (!running) listModels()
    }

    private fun modelFiles(): List<File> =
        listOfNotNull(filesDir, getExternalFilesDir(null))
            .flatMap { it.listFiles { f -> f.name.endsWith(".gguf") }.orEmpty().toList() }
            .sortedBy { it.name }

    private fun listModels() {
        val files = modelFiles()
        models.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, files.map { it.name })
        go.isEnabled = files.isNotEmpty()
        if (files.isEmpty()) status.text = "No .gguf model found. Push one to ${getExternalFilesDir(null)} (see android/README.md)."
    }

    private fun start() {
        val file = modelFiles().getOrNull(models.selectedItemPosition) ?: return
        val coreList = cores.text.toString().trim()
        val text = prompt.text.toString()
        val asChat = chat.isChecked
        val draft = if (lookup.isChecked) 3 else 0
        val where = BACKENDS[backend.selectedItemPosition]
        running = true
        stop.set(false)
        go.text = "Stop"
        output.text = ""
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        engine.execute {
            try {
                val key = "${file.path}|$where|$coreList"
                if (session == null || loaded != key) {
                    session?.close()
                    session = null
                    ui { status.text = "Loading ${file.name}…" }
                    val t0 = System.nanoTime()
                    session = when (where) {
                        "GPU" -> Aether.openGpu(file.path, coreList, MAX_SEQ)
                        "DSP" -> Aether.openDsp(file.path, filesDir.path, MAX_SEQ)
                        else -> Aether.open(file.path, coreList, MAX_SEQ)
                    }
                    loaded = key
                    val ms = (System.nanoTime() - t0) / 1e6
                    val on = when (where) {
                        "GPU" -> "GPU, prompt on CPU $coreList"
                        "DSP" -> "Hexagon DSP"
                        else -> "CPU $coreList"
                    }
                    ui { status.text = "Loaded in %.0f ms (%s)".format(ms, on) }
                }
                session!!.setLookup(draft)
                val utf8 = Utf8Stream()
                val st = session!!.generate(text, asChat, 0.7f, 0.9f, MAX_TOKENS, System.nanoTime()) { bytes ->
                    val piece = utf8.feed(bytes)
                    ui {
                        output.append(piece)
                        scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
                    }
                    !stop.get()
                }
                ui { status.text = describe(st) }
            } catch (e: Exception) {
                ui { status.text = "Error: ${e.message}" }
            } finally {
                ui {
                    running = false
                    go.text = "Generate"
                    window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                }
            }
        }
    }

    private fun describe(s: Aether.Stats): String {
        val pre = "prefill %d tok in %.0f ms (%.1f tok/s)".format(s.prefillTokens, s.prefillMs, s.prefillTokens * 1e3 / s.prefillMs)
        if (s.decodeTokens == 0) return pre
        return pre + " · decode %d tok at %.1f tok/s".format(s.decodeTokens, s.decodeTokens * 1e3 / s.decodeMs)
    }

    private fun ui(block: () -> Unit) = runOnUiThread(block)

    override fun onDestroy() {
        stop.set(true)
        engine.execute { session?.close() }
        engine.shutdown()
        super.onDestroy()
    }

    companion object {
        /** KV cache length: prompt plus reply, in tokens. */
        const val MAX_SEQ = 2048
        const val MAX_TOKENS = 512
        val BACKENDS = listOf("CPU", "GPU", "DSP")
    }
}
