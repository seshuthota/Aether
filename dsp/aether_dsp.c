// Our library on the Hexagon DSP (cDSP): the calls in aether_dsp.idl. The
// CPU opens it with remote_handle64_open (src/kernels/hexagon.rs); FastRPC
// then calls aether_dsp_open here, and every later call arrives through
// the qaic-generated aether_dsp_skel.c.

#include <stdlib.h>
#include <string.h>

#include "AEEStdErr.h"
#include "HAP_power.h"
#include "aether_dsp.h"
#include "hexagon_types.h"
#include "hvx_hexagon_protos.h"

// One HVX vector: 128 bytes, 32 floats.
#define VF 32

struct ctx {
    int unused;
};

AEEResult aether_dsp_open(const char *uri, remote_handle64 *h) {
    (void) uri;
    struct ctx *c = calloc(1, sizeof(*c));
    if (!c) return AEE_ENOMEMORY;
    *h = (remote_handle64) c;
    // A compute client, with the vector units powered up. (HVX contexts
    // themselves are handed to a thread by QuRT on first use.)
    HAP_power_request_t r;
    memset(&r, 0, sizeof(r));
    r.type = HAP_power_set_apptype;
    r.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    int err = HAP_power_set(c, &r);
    if (err) return err;
    // Clocks: core and bus at their highest voltage corner, and no sleep
    // between calls (else every call wakes a sleeping DSP).
    memset(&r, 0, sizeof(r));
    r.type = HAP_power_set_DCVS_v3;
    r.dcvs_v3.set_dcvs_enable = TRUE;
    r.dcvs_v3.dcvs_enable = TRUE;
    r.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
    r.dcvs_v3.set_bus_params = TRUE;
    r.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.set_core_params = TRUE;
    r.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.set_sleep_disable = TRUE;
    r.dcvs_v3.sleep_disable = TRUE;
    err = HAP_power_set(c, &r);
    if (err) return err;
    memset(&r, 0, sizeof(r));
    r.type = HAP_power_set_HVX;
    r.hvx.power_up = TRUE;
    return HAP_power_set(c, &r);
}

AEEResult aether_dsp_close(remote_handle64 h) {
    free((void *) h);
    return AEE_SUCCESS;
}

AEEResult aether_dsp_ping(remote_handle64 h, uint32 x, uint32 *y) {
    (void) h;
    *y = x + 1;
    return AEE_SUCCESS;
}

// c = a + b, 32 lanes at a time; the tail (n % 32) one by one. FastRPC's
// copies of the buffers need not be 128-byte aligned, so the vectors are
// loaded and stored unaligned.
AEEResult aether_dsp_add(remote_handle64 h, const float *a, int an, const float *b, int bn, float *c, int cn) {
    (void) h;
    if (an != bn || an != cn) return AEE_EBADPARM;
    int i = 0;
    for (; i + VF <= an; i += VF) {
        HVX_Vector va = *(const HVX_UVector *) (a + i);
        HVX_Vector vb = *(const HVX_UVector *) (b + i);
        *(HVX_UVector *) (c + i) = Q6_Vsf_vadd_VsfVsf(va, vb);
    }
    for (; i < an; i++) c[i] = a[i] + b[i];
    return AEE_SUCCESS;
}
