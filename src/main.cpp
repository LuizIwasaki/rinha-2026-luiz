// Rinha de Backend 2026 - Fraud Detection API (IVF + SSE2)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <emmintrin.h>

static constexpr int DIMS = 14;
static constexpr int KNN = 5;
static constexpr float THRESHOLD = 0.6f;
static constexpr int NPROBE = 25;
static constexpr int MAX_EVENTS = 512;
static constexpr int BUF_SIZE = 8192;

static constexpr float MAX_AMOUNT = 10000.0f;
static constexpr float MAX_INST = 12.0f;
static constexpr float AVG_RATIO = 10.0f;
static constexpr float MAX_MIN = 1440.0f;
static constexpr float MAX_KM = 1000.0f;
static constexpr float MAX_TX24 = 20.0f;
static constexpr float MAX_MAVG = 10000.0f;

static int32_t g_nrefs = 0;
static int32_t g_ncent = 0;
static float*  g_centroids = nullptr;
static int32_t* g_offsets = nullptr;
static float*  g_vecs  = nullptr;
static uint8_t* g_lbls = nullptr;

static float mcc_risk(const char* m, int len) {
    if (len != 4) return 0.5f;
    int v = (m[0]-'0')*1000+(m[1]-'0')*100+(m[2]-'0')*10+(m[3]-'0');
    switch(v) {
        case 5411: return 0.15f; case 5812: return 0.30f;
        case 5912: return 0.20f; case 5944: return 0.45f;
        case 7801: return 0.80f; case 7802: return 0.75f;
        case 7995: return 0.85f; case 4511: return 0.35f;
        case 5311: return 0.25f; case 5999: return 0.50f;
    }
    return 0.5f;
}

static void* map_file(const char* path, size_t& sz) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); return nullptr; }
    struct stat st; fstat(fd, &st); sz = st.st_size;
    void* p = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE|MAP_POPULATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return nullptr;
    return p;
}

static bool load_data() {
    const char* dir = getenv("DATA_DIR");
    if (!dir) dir = "/data";
    char p[512]; size_t sz;

    snprintf(p, sizeof(p), "%s/centroids.bin", dir);
    auto* cm = (uint8_t*)map_file(p, sz);
    if (!cm) return false;
    g_ncent = *(int32_t*)cm;
    g_centroids = (float*)(cm + 8);
    fprintf(stderr, "Loaded %d centroids\n", g_ncent);

    snprintf(p, sizeof(p), "%s/offsets.bin", dir);
    auto* om = (uint8_t*)map_file(p, sz);
    if (!om) return false;
    g_offsets = (int32_t*)(om + 4);

    snprintf(p, sizeof(p), "%s/vectors.bin", dir);
    auto* vm = (uint8_t*)map_file(p, sz);
    if (!vm) return false;
    g_nrefs = *(int32_t*)vm;
    g_vecs = (float*)(vm + 4);
    fprintf(stderr, "Loaded %d vectors\n", g_nrefs);

    snprintf(p, sizeof(p), "%s/labels.bin", dir);
    auto* lm = (uint8_t*)map_file(p, sz);
    if (!lm) return false;
    g_lbls = lm + 4;
    return true;
}

static inline float dist_sq_sse(const float* __restrict__ a, const float* __restrict__ b) {
    __m128 va = _mm_loadu_ps(a);
    __m128 vb = _mm_loadu_ps(b);
    __m128 d = _mm_sub_ps(va, vb);
    __m128 sum = _mm_mul_ps(d, d);
    va = _mm_loadu_ps(a + 4); vb = _mm_loadu_ps(b + 4);
    d = _mm_sub_ps(va, vb); sum = _mm_add_ps(sum, _mm_mul_ps(d, d));
    va = _mm_loadu_ps(a + 8); vb = _mm_loadu_ps(b + 8);
    d = _mm_sub_ps(va, vb); sum = _mm_add_ps(sum, _mm_mul_ps(d, d));
    __m128 shuf = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 3, 0, 1));
    sum = _mm_add_ps(sum, shuf);
    shuf = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 0, 3, 2));
    sum = _mm_add_ps(sum, shuf);
    float result; _mm_store_ss(&result, sum);
    float d12 = a[12]-b[12], d13 = a[13]-b[13];
    return result + d12*d12 + d13*d13;
}

static const char* find_key(const char* js, const char* key) { return strstr(js, key); }
static double get_num(const char* js, const char* key) {
    const char* p = find_key(js, key); if (!p) return 0.0; p += strlen(key);
    while (*p && (*p=='"'||*p==':'||*p==' '||*p=='\t')) p++; return strtod(p, nullptr);
}
static bool get_bool(const char* js, const char* key) {
    const char* p = find_key(js, key); if (!p) return false; p += strlen(key);
    while (*p && (*p=='"'||*p==':'||*p==' ')) p++; return (*p == 't');
}
static bool has_null(const char* js, const char* key) {
    const char* p = find_key(js, key); if (!p) return true; p += strlen(key);
    while (*p && (*p=='"'||*p==':'||*p==' ')) p++; return (*p == 'n');
}
static int get_mcc(const char* js, char* out) {
    const char* p = find_key(js, "\"mcc\""); if (!p) { out[0]=0; return 0; } p += 5;
    while (*p && *p != '"') p++; if (*p == '"') p++;
    int i = 0; while (*p && *p != '"' && i < 8) out[i++] = *p++; out[i] = 0; return i;
}
static bool is_known_merchant(const char* js) {
    const char* mid_key = find_key(js, "\"merchant\""); if (!mid_key) return false;
    const char* mid_p = find_key(mid_key, "\"id\""); if (!mid_p) return false; mid_p += 4;
    while (*mid_p && *mid_p != '"') mid_p++; if (*mid_p == '"') mid_p++;
    const char* mid_end = mid_p; while (*mid_end && *mid_end != '"') mid_end++;
    int mid_len = mid_end - mid_p;
    const char* km = find_key(js, "\"known_merchants\""); if (!km) return false;
    km = strchr(km, '['); if (!km) return false; const char* end = strchr(km, ']'); if (!end) return false;
    const char* s = km; while (s < end) {
        s = strchr(s, '"'); if (!s || s >= end) break; s++;
        const char* se = strchr(s, '"'); if (!se) break;
        if (se - s == mid_len && memcmp(s, mid_p, mid_len) == 0) return true;
        s = se + 1;
    }
    return false;
}
static int get_hour(const char* js, const char* key) {
    const char* p = find_key(js, key); if (!p) return 0; p += strlen(key);
    const char* t = strchr(p, 'T'); if (!t) return 0; return (t[1]-'0')*10 + (t[2]-'0');
}
static int get_dow(const char* js, const char* key) {
    const char* p = find_key(js, key); if (!p) return 0; p += strlen(key);
    while (*p && (*p=='"'||*p==':'||*p==' ')) p++; if (*p == '"') p++;
    int y=(p[0]-'0')*1000+(p[1]-'0')*100+(p[2]-'0')*10+(p[3]-'0');
    int m=(p[5]-'0')*10+(p[6]-'0'), d=(p[8]-'0')*10+(p[9]-'0');
    static int t[]={0,3,2,5,0,3,5,1,4,6,2,4}; if (m<3) y--;
    int dow=(y+y/4-y/100+y/400+t[m-1]+d)%7; return dow==0?6:dow-1;
}
static double get_minutes_since(const char* js) {
    const char* lt = find_key(js, "\"last_transaction\""); if (!lt) return -1;
    const char* ts = find_key(lt, "\"timestamp\""); if (!ts) return -1;
    ts += 11; while (*ts && (*ts=='"'||*ts==':'||*ts==' ')) ts++; if (*ts == '"') ts++;
    const char* ra = find_key(js, "\"requested_at\""); if (!ra) return -1;
    ra += 14; while (*ra && (*ra=='"'||*ra==':'||*ra==' ')) ra++; if (*ra == '"') ra++;
    auto parse_ts = [](const char* s, int& Y, int& M, int& D, int& h, int& m, int& sec) {
        Y=(s[0]-'0')*1000+(s[1]-'0')*100+(s[2]-'0')*10+(s[3]-'0');
        M=(s[5]-'0')*10+(s[6]-'0'); D=(s[8]-'0')*10+(s[9]-'0');
        h=(s[11]-'0')*10+(s[12]-'0'); m=(s[14]-'0')*10+(s[15]-'0');
        sec=(s[17]-'0')*10+(s[18]-'0');
    };
    int y1,m1,d1,h1,mi1,s1,y2,m2,d2,h2,mi2,s2;
    parse_ts(ra, y1,m1,d1,h1,mi1,s1); parse_ts(ts, y2,m2,d2,h2,mi2,s2);
    auto to_min=[](int y,int m,int d,int h,int mi,int)->long long{
        long long days=(long long)y*365+y/4-y/100+y/400;
        int md[]={0,31,59,90,120,151,181,212,243,273,304,334};
        days+=md[m-1]+d; if(m>2&&((y%4==0&&y%100!=0)||y%400==0)) days++;
        return days*1440LL+h*60LL+mi;
    };
    return (double)(to_min(y1,m1,d1,h1,mi1,s1)-to_min(y2,m2,d2,h2,mi2,s2));
}
static inline float clamp01(float x) { return x<0.0f?0.0f:(x>1.0f?1.0f:x); }

static void vectorize(const char* js, float vec[DIMS]) {
    float amount = get_num(js, "\"amount\"");
    float installments = get_num(js, "\"installments\"");
    float avg_amount = get_num(js, "\"avg_amount\"");
    float tx_count = get_num(js, "\"tx_count_24h\"");
    float km_home = get_num(js, "\"km_from_home\"");
    bool is_online = get_bool(js, "\"is_online\"");
    bool card_present = get_bool(js, "\"card_present\"");
    bool known = is_known_merchant(js);
    char mcc[16]; int mcc_len = get_mcc(js, mcc);
    float merch_avg = 0;
    const char* merch = find_key(js, "\"merchant\"");
    if (merch) {
        const char* ma = find_key(merch, "\"avg_amount\"");
        if (ma) { ma+=12; while(*ma&&(*ma=='"'||*ma==':'||*ma==' '))ma++; merch_avg=strtof(ma,nullptr); }
    }
    bool last_tx_null = has_null(js, "\"last_transaction\"");
    vec[0] = clamp01(amount / MAX_AMOUNT); vec[1] = clamp01(installments / MAX_INST);
    vec[2] = avg_amount>0 ? clamp01((amount/avg_amount)/AVG_RATIO) : 0.0f;
    vec[3] = (float)get_hour(js,"\"requested_at\"")/23.0f;
    vec[4] = (float)get_dow(js,"\"requested_at\"")/6.0f;
    vec[5] = last_tx_null ? -1.0f : clamp01((float)(get_minutes_since(js)/MAX_MIN));
    vec[6] = last_tx_null ? -1.0f : clamp01((float)get_num(js,"\"km_from_current\"")/MAX_KM);
    vec[7] = clamp01(km_home / MAX_KM); vec[8] = clamp01(tx_count / MAX_TX24);
    vec[9] = is_online ? 1.0f : 0.0f; vec[10] = card_present ? 1.0f : 0.0f;
    vec[11] = known ? 0.0f : 1.0f; vec[12] = mcc_risk(mcc, mcc_len); vec[13] = clamp01(merch_avg / MAX_MAVG);
}

struct Neighbor { float dist; int idx; };
static inline void heap_down(Neighbor* h, int n, int i) {
    while (2*i+1 < n) {
        int c = 2*i+1; if (c+1<n && h[c+1].dist>h[c].dist) c++;
        if (h[i].dist >= h[c].dist) break;
        Neighbor t=h[i]; h[i]=h[c]; h[c]=t; i=c;
    }
}
static void scan_range(const float query[DIMS], int start, int end, Neighbor* top, int& n_top) {
    for (int i = start; i < end; i++) {
        float d = dist_sq_sse(query, g_vecs + i * DIMS);
        if (n_top < KNN) {
            top[n_top++] = {d, i}; if (n_top == KNN) for (int j = KNN/2-1; j >= 0; j--) heap_down(top, KNN, j);
        } else if (d < top[0].dist) {
            top[0] = {d, i}; heap_down(top, KNN, 0);
        }
    }
}
static float knn_search(const float query[DIMS]) {
    struct CDist { float d; int idx; }; CDist probe[NPROBE]; int n_probe = 0;
    for (int c = 0; c < g_ncent; c++) {
        float d = dist_sq_sse(query, g_centroids + c * DIMS);
        if (n_probe < NPROBE) {
            probe[n_probe++] = {d, c};
            if (n_probe == NPROBE) {
                for (int j = NPROBE/2-1; j >= 0; j--) {
                    int i = j;
                    while (2*i+1<NPROBE) {
                        int ch=2*i+1; if(ch+1<NPROBE && probe[ch+1].d>probe[ch].d) ch++;
                        if(probe[i].d>=probe[ch].d) break;
                        CDist t=probe[i]; probe[i]=probe[ch]; probe[ch]=t; i=ch;
                    }
                }
            }
        } else if (d < probe[0].d) {
            probe[0] = {d, c};
            int i = 0;
            while (2*i+1<NPROBE) {
                int ch=2*i+1; if(ch+1<NPROBE && probe[ch+1].d>probe[ch].d) ch++;
                if(probe[i].d>=probe[ch].d) break;
                CDist t=probe[i]; probe[i]=probe[ch]; probe[ch]=t; i=ch;
            }
        }
    }
    Neighbor top[KNN]; int n_top = 0;
    for (int p = 0; p < n_probe; p++) {
        int ci = probe[p].idx; scan_range(query, g_offsets[ci], g_offsets[ci + 1], top, n_top);
    }
    float worst = (n_top == KNN) ? top[0].dist : 1e30f;
    if (worst > 1.0f) scan_range(query, 0, g_nrefs, top, n_top);
    int frauds = 0; for (int i = 0; i < n_top; i++) if (g_lbls[top[i].idx] == 1) frauds++;
    return (float)frauds / (float)KNN;
}

static int process_request(const char* req, int reqlen, char* resp) {
    if (reqlen < 5) return 0;
    if (memcmp(req, "GET", 3) == 0 && strstr(req, "/ready")) {
        return snprintf(resp, BUF_SIZE, "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
    }
    if (memcmp(req, "POST", 4) == 0 && strstr(req, "/fraud-score")) {
        const char* body = strstr(req, "\r\n\r\n"); if (!body) return 0; body += 4;
        float query[DIMS]; vectorize(body, query); float score = knn_search(query); bool approved = score < THRESHOLD;
        char json[128]; int jlen = snprintf(json, sizeof(json), "{\"approved\":%s,\"fraud_score\":%.1f}", approved ? "true" : "false", score);
        return snprintf(resp, BUF_SIZE, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s", jlen, json);
    }
    return snprintf(resp, BUF_SIZE, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
}

struct ConnBuf { char buf[BUF_SIZE]; int len; };
static ConnBuf g_conns[65536];

int main() {
    int port = 8080;
    const char* pe = getenv("PORT"); if (pe) port = atoi(pe);
    const char* sock_path = getenv("SOCKET_PATH");
    if (!load_data()) { fprintf(stderr, "Failed to load data\n"); return 1; }

    int sfd;
    if (sock_path) {
        sfd = socket(AF_UNIX, SOCK_STREAM, 0);
        unlink(sock_path);
        struct sockaddr_un addr = {};
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path)-1);
        if (bind(sfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind unix"); return 1; }
        chmod(sock_path, 0666);
        fprintf(stderr, "Server ready on unix socket %s\n", sock_path);
    } else {
        sfd = socket(AF_INET, SOCK_STREAM, 0);
        int opt = 1; setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR|SO_REUSEPORT, &opt, sizeof(opt));
        struct sockaddr_in addr = {}; addr.sin_family = AF_INET; addr.sin_addr.s_addr = INADDR_ANY; addr.sin_port = htons(port);
        if (bind(sfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind tcp"); return 1; }
        fprintf(stderr, "Server ready on port %d\n", port);
    }

    listen(sfd, SOMAXCONN);
    fcntl(sfd, F_SETFL, fcntl(sfd, F_GETFL, 0) | O_NONBLOCK);
    int epfd = epoll_create1(0);
    struct epoll_event ev; ev.events = EPOLLIN; ev.data.fd = sfd;
    epoll_ctl(epfd, EPOLL_CTL_ADD, sfd, &ev);

    struct epoll_event events[MAX_EVENTS];
    char resp[BUF_SIZE];
    while (true) {
        int n = epoll_wait(epfd, events, MAX_EVENTS, -1);
        for (int i = 0; i < n; i++) {
            int fd = events[i].data.fd;
            if (fd == sfd) {
                while (true) {
                    int cfd = accept4(sfd, nullptr, nullptr, SOCK_NONBLOCK);
                    if (cfd < 0) break;
                    if (!sock_path) { int nd=1; setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &nd, sizeof(nd)); }
                    g_conns[cfd].len = 0;
                    ev.events = EPOLLIN | EPOLLET; ev.data.fd = cfd;
                    epoll_ctl(epfd, EPOLL_CTL_ADD, cfd, &ev);
                }
            } else {
                ConnBuf& cb = g_conns[fd];
                bool closed = false;
                while (true) {
                    int r = read(fd, cb.buf+cb.len, BUF_SIZE-cb.len-1);
                    if (r == 0) { closed = true; break; }
                    if (r < 0) { if (errno != EAGAIN) closed = true; break; }
                    cb.len += r;
                }
                if (closed) { epoll_ctl(epfd, EPOLL_CTL_DEL, fd, nullptr); close(fd); continue; }

                cb.buf[cb.len] = 0;
                const char* hdr_end = strstr(cb.buf, "\r\n\r\n");
                if (!hdr_end) {
                    if (cb.len >= BUF_SIZE-1) { epoll_ctl(epfd,EPOLL_CTL_DEL,fd,nullptr); close(fd); }
                    continue;
                }

                bool complete = true;
                if (memcmp(cb.buf, "POST", 4) == 0) {
                    const char* cl = strstr(cb.buf, "Content-Length:");
                    if (!cl) cl = strstr(cb.buf, "content-length:");
                    if (cl) {
                        cl += 15; int clen = atoi(cl); int bs = (hdr_end+4)-cb.buf;
                        if (cb.len-bs < clen) complete = false;
                    }
                }
                if (!complete) continue;

                int rlen = process_request(cb.buf, cb.len, resp);
                if (rlen > 0) {
                    int sent = 0;
                    while (sent < rlen) {
                        int w = write(fd, resp+sent, rlen-sent);
                        if (w <= 0) break;
                        sent += w;
                    }
                }
                epoll_ctl(epfd, EPOLL_CTL_DEL, fd, nullptr);
                close(fd);
            }
        }
    }
    return 0;
}
