/* JNI shim: exposes include/aether.h to dev.aether.chat.Aether.
 *
 * Strings arrive as byte[] of UTF-8 (Kotlin's String.toByteArray()), not
 * jstring: JNI's GetStringUTFChars gives "modified UTF-8", which encodes
 * emoji and NUL differently from the real UTF-8 the engine expects.
 * Errors become a java.io.IOException carrying aether_last_error(). */
#include <jni.h>
#include <stdlib.h>
#include <string.h>
#include "aether.h"

/* A NUL-terminated copy of a byte[] (caller frees), or NULL for null. */
static char *cstr(JNIEnv *env, jbyteArray a) {
    if (!a) return NULL;
    jsize n = (*env)->GetArrayLength(env, a);
    char *s = malloc((size_t)n + 1);
    (*env)->GetByteArrayRegion(env, a, 0, n, (jbyte *)s);
    s[n] = 0;
    return s;
}

static void throw_error(JNIEnv *env) {
    jclass c = (*env)->FindClass(env, "java/io/IOException");
    (*env)->ThrowNew(env, c, aether_last_error());
}

JNIEXPORT jlong JNICALL
Java_dev_aether_chat_Aether_nativeOpen(JNIEnv *env, jclass cls, jbyteArray path, jbyteArray cores, jint max_seq) {
    (void)cls;
    char *p = cstr(env, path), *c = cstr(env, cores);
    AetherSession *s = NULL;
    int code = aether_open(p, c, (uint32_t)max_seq, &s);
    free(p);
    free(c);
    if (code != AETHER_OK) {
        throw_error(env);
        return 0;
    }
    return (jlong)(intptr_t)s;
}

/* What on_text needs to call back into Kotlin. The engine calls on_text on
 * the thread that called aether_generate, so `env` stays valid. */
struct callback {
    JNIEnv *env;
    jobject target;
    jmethodID on_text;
};

static int on_text(const uint8_t *bytes, size_t len, void *user) {
    struct callback *cb = user;
    JNIEnv *env = cb->env;
    if (len == 0) return 1;
    jbyteArray a = (*env)->NewByteArray(env, (jsize)len);
    (*env)->SetByteArrayRegion(env, a, 0, (jsize)len, (const jbyte *)bytes);
    jboolean go = (*env)->CallBooleanMethod(env, cb->target, cb->on_text, a);
    (*env)->DeleteLocalRef(env, a);
    /* A Kotlin exception stops generation; it is rethrown on return. */
    return !(*env)->ExceptionCheck(env) && go;
}

/* Returns {prefill tokens, prefill ms, decode tokens, decode ms}. */
JNIEXPORT jdoubleArray JNICALL
Java_dev_aether_chat_Aether_nativeGenerate(JNIEnv *env, jclass cls, jlong handle, jbyteArray prompt, jboolean chat,
                                           jfloat temperature, jfloat top_p, jint max_tokens, jlong seed, jobject target) {
    (void)cls;
    jclass tc = (*env)->GetObjectClass(env, target);
    struct callback cb = {env, target, (*env)->GetMethodID(env, tc, "onText", "([B)Z")};
    char *p = cstr(env, prompt);
    AetherStats st = {0};
    int code = aether_generate((AetherSession *)(intptr_t)handle, p, chat, temperature, top_p, (uint32_t)max_tokens,
                               (uint64_t)seed, on_text, &cb, &st);
    free(p);
    if ((*env)->ExceptionCheck(env)) return NULL;
    if (code != AETHER_OK) {
        throw_error(env);
        return NULL;
    }
    jdouble v[4] = {st.prefill_tokens, st.prefill_ms, st.decode_tokens, st.decode_ms};
    jdoubleArray out = (*env)->NewDoubleArray(env, 4);
    (*env)->SetDoubleArrayRegion(env, out, 0, 4, v);
    return out;
}

JNIEXPORT void JNICALL
Java_dev_aether_chat_Aether_nativeSetLookup(JNIEnv *env, jclass cls, jlong handle, jint k, jint ngram) {
    (void)cls;
    if (aether_set_lookup((AetherSession *)(intptr_t)handle, (uint32_t)k, (uint32_t)ngram) != AETHER_OK) throw_error(env);
}

JNIEXPORT void JNICALL
Java_dev_aether_chat_Aether_nativeClose(JNIEnv *env, jclass cls, jlong handle) {
    (void)env;
    (void)cls;
    aether_close((AetherSession *)(intptr_t)handle);
}
