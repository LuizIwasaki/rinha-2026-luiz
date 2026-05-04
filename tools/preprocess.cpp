// IVF preprocessor: JSON -> k-means clustered binary files
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

static const int DIMS = 14;
static const int K = 2048;
static const int KMEANS_ITER = 8;

static float dist_sq(const float* a, const float* b) {
    float s = 0;
    for (int d = 0; d < DIMS; d++) { float x = a[d]-b[d]; s += x*x; }
    return s;
}

int main(int argc, char* argv[]) {
    const char* input   = argc > 1 ? argv[1] : "/data/references.json";
    const char* vec_out = argc > 2 ? argv[2] : "/data/vectors.bin";
    const char* lbl_out = argc > 3 ? argv[3] : "/data/labels.bin";
    const char* cen_out = argc > 4 ? argv[4] : "/data/centroids.bin";
    const char* off_out = argc > 5 ? argv[5] : "/data/offsets.bin";

    FILE* f = fopen(input, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", input); return 1; }
    fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    fprintf(stderr, "Reading %s (%ld MB)...\n", input, fsize/(1024*1024));
    char* data = (char*)malloc(fsize + 1);
    fread(data, 1, fsize, f); data[fsize] = 0; fclose(f);

    std::vector<float> vecs;
    std::vector<uint8_t> lbls;
    vecs.reserve(3000000 * DIMS);
    lbls.reserve(3000000);

    char* p = data; int count = 0;
    while ((p = strstr(p, "\"vector\"")) != nullptr) {
        p = strchr(p, '['); if (!p) break; p++;
        for (int i = 0; i < DIMS; i++) {
            while (*p==' '||*p==','||*p=='\n'||*p=='\r'||*p=='\t') p++;
            char* end; vecs.push_back(strtof(p, &end)); p = end;
        }
        char* lbl = strstr(p, "\"label\""); if (!lbl) break;
        char* q = strchr(strchr(lbl, ':'), '"'); if (!q) break; q++;
        lbls.push_back(*q == 'f' ? 1 : 0);
        p = q; count++;
        if (count % 500000 == 0) fprintf(stderr, "  parsed %d...\n", count);
    }
    free(data);
    fprintf(stderr, "Parsed %d vectors\n", count);

    // K-means clustering
    std::vector<float> centroids(K * DIMS);
    std::vector<int32_t> assignments(count);

    std::mt19937 rng(42);
    std::vector<int> indices(count);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    for (int i = 0; i < K; i++)
        memcpy(&centroids[i * DIMS], &vecs[indices[i] * DIMS], DIMS * sizeof(float));

    fprintf(stderr, "K-means K=%d, %d iters...\n", K, KMEANS_ITER);

    for (int iter = 0; iter < KMEANS_ITER; iter++) {
        #pragma omp parallel for schedule(dynamic, 1024)
        for (int i = 0; i < count; i++) {
            const float* v = &vecs[i * DIMS];
            float best_d = 1e30f; int best_c = 0;
            for (int c = 0; c < K; c++) {
                float d = dist_sq(v, &centroids[c * DIMS]);
                if (d < best_d) { best_d = d; best_c = c; }
            }
            assignments[i] = best_c;
        }

        std::vector<double> sums(K * DIMS, 0.0);
        std::vector<int> cnts(K, 0);
        for (int i = 0; i < count; i++) {
            int c = assignments[i];
            cnts[c]++;
            const float* v = &vecs[i * DIMS];
            double* s = &sums[c * DIMS];
            for (int d = 0; d < DIMS; d++) s[d] += v[d];
        }
        for (int c = 0; c < K; c++) {
            if (cnts[c] == 0) continue;
            for (int d = 0; d < DIMS; d++)
                centroids[c * DIMS + d] = (float)(sums[c * DIMS + d] / cnts[c]);
        }

        int mn = count, mx = 0;
        for (int c = 0; c < K; c++) { mn = std::min(mn, cnts[c]); mx = std::max(mx, cnts[c]); }
        fprintf(stderr, "  iter %d: min=%d max=%d avg=%d\n", iter, mn, mx, count/K);
    }

    // Sort vectors by cluster assignment
    std::vector<uint32_t> idx(count);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
        return assignments[a] < assignments[b];
    });

    std::vector<int32_t> offsets(K + 1, 0);
    for (int i = 0; i < count; i++) offsets[assignments[idx[i]] + 1]++;
    for (int i = 1; i <= K; i++) offsets[i] += offsets[i-1];

    FILE* cf = fopen(cen_out, "wb");
    int32_t k = K, dims = DIMS;
    fwrite(&k, 4, 1, cf); fwrite(&dims, 4, 1, cf);
    fwrite(centroids.data(), sizeof(float), K * DIMS, cf);
    fclose(cf);

    FILE* of = fopen(off_out, "wb");
    fwrite(&k, 4, 1, of);
    fwrite(offsets.data(), sizeof(int32_t), K + 1, of);
    fclose(of);

    FILE* vf = fopen(vec_out, "wb");
    int32_t n = count;
    fwrite(&n, 4, 1, vf);
    for (int i = 0; i < count; i++)
        fwrite(&vecs[idx[i] * DIMS], sizeof(float), DIMS, vf);
    fclose(vf);

    FILE* lf = fopen(lbl_out, "wb");
    fwrite(&n, 4, 1, lf);
    for (int i = 0; i < count; i++) {
        uint8_t lb = lbls[idx[i]];
        fwrite(&lb, 1, 1, lf);
    }
    fclose(lf);

    fprintf(stderr, "Done. K=%d centroids, %d vectors\n", K, count);
    return 0;
}
