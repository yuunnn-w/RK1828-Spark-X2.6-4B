// SPDX-FileCopyrightText: 2026 yuunnn-w
// SPDX-License-Identifier: GPL-3.0-only
// 本文件属 RK1828-Spark-X2.6-4B 项目；许可证见 LICENSE（GPL-3.0）与 NOTICE（第三方部件）。
// httpd.cpp — HTTP/1.1 实现（keep-alive、超时、连接上限、SSE）
#include "httpd.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static const char* status_text(int code) {
  switch (code) {
    case 200: return "OK";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 413: return "Payload Too Large";
    case 429: return "Too Many Requests";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 503: return "Service Unavailable";
    default: return "Unknown";
  }
}

static void set_sock_timeout(int fd, int send_sec, int recv_sec) {
  struct timeval tv;
  tv.tv_sec = recv_sec;
  tv.tv_usec = 0;
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  tv.tv_sec = send_sec;
  tv.tv_usec = 0;
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

class SocketWriter : public ResponseWriter {
 public:
  SocketWriter(int fd, bool keep_alive) : m_fd(fd), m_keep_alive(keep_alive) {}

  bool write_raw(const char* p, size_t n) {
    while (n > 0) {
      ssize_t w = write(m_fd, p, n);
      if (w <= 0) {
        if (w < 0 && errno == EINTR) {
          continue;
        }
        m_alive = false;
        return false;
      }
      p += w;
      n -= (size_t)w;
    }
    return true;
  }

  void write_full(int status, const std::string& body, const std::string& ctype) override {
    std::string h = "HTTP/1.1 " + std::to_string(status) + " " + status_text(status) + "\r\n";
    h += "Content-Type: " + ctype + "\r\n";
    h += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    h += "Access-Control-Allow-Origin: *\r\n";
    if (status == 429) {
      h += "Retry-After: 2\r\n";
    } else if (status == 503) {
      h += "Retry-After: 5\r\n";
    }
    h += m_keep_alive ? "Connection: keep-alive\r\n" : "Connection: close\r\n";
    h += "\r\n";
    write_raw(h.data(), h.size());
    if (!body.empty()) {
      write_raw(body.data(), body.size());
    }
  }

  void write_status_only(int status, const std::string& ctype) override {
    std::string h = "HTTP/1.1 " + std::to_string(status) + " " + status_text(status) + "\r\n";
    h += "Content-Type: " + ctype + "\r\n";
    h += "Access-Control-Allow-Origin: *\r\n";
    h += m_keep_alive ? "Connection: keep-alive\r\n" : "Connection: close\r\n";
    h += "\r\n";
    write_raw(h.data(), h.size());
  }

  // 流式响应必须结束这条连接（没有 Content-Length，只能靠关连接界定边界）
  bool write_sse_start() override {
    std::string h = "HTTP/1.1 200 OK\r\n";
    h += "Content-Type: text/event-stream; charset=utf-8\r\n";
    h += "Cache-Control: no-cache, no-transform\r\n";
    h += "X-Accel-Buffering: no\r\n";
    h += "Access-Control-Allow-Origin: *\r\n";
    h += "Connection: close\r\n\r\n";
    m_keep_alive = false;
    return write_raw(h.data(), h.size());
  }

  bool write_sse_data(const std::string& payload) override {
    std::string s = "data: " + payload + "\n\n";
    return write_raw(s.data(), s.size());
  }

  void write_sse_end() override {
    static const char* done = "data: [DONE]\n\n";
    write_raw(done, strlen(done));
  }

  bool alive() override { return m_alive; }
  bool keep_alive() const { return m_keep_alive; }

 private:
  int m_fd;
  bool m_alive = true;
  bool m_keep_alive;
};

HttpServer::HttpServer() : m_listen_fd(-1), m_running(false), m_stop_requested(false), m_active(0), m_total(0) {}

HttpServer::~HttpServer() { stop(); }

int HttpServer::start(const HttpServerOptions& opt, HttpHandler handler) {
  m_opt = opt;
  m_handler = handler;

  signal(SIGPIPE, SIG_IGN);

  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    fprintf(stderr, "socket failed: %s\n", strerror(errno));
    return -1;
  }
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  struct sockaddr_in a;
  memset(&a, 0, sizeof(a));
  a.sin_family = AF_INET;
  a.sin_port = htons((uint16_t)m_opt.port);
  if (inet_pton(AF_INET, m_opt.host.c_str(), &a.sin_addr) <= 0) {
    a.sin_addr.s_addr = INADDR_ANY;
  }
  if (bind(fd, (struct sockaddr*)&a, sizeof(a)) != 0) {
    fprintf(stderr, "bind %s:%d failed: %s\n", m_opt.host.c_str(), m_opt.port, strerror(errno));
    close(fd);
    return -1;
  }
  if (listen(fd, m_opt.backlog) != 0) {
    fprintf(stderr, "listen failed: %s\n", strerror(errno));
    close(fd);
    return -1;
  }
  m_listen_fd = fd;
  m_running = true;
  m_stop_requested = false;
  std::thread t(&HttpServer::accept_loop, this);
  t.detach();
  return 0;
}

void HttpServer::stop() {
  m_stop_requested = true;
  if (m_listen_fd >= 0) {
    shutdown(m_listen_fd, SHUT_RDWR);
    close(m_listen_fd);
    m_listen_fd = -1;
  }
  m_running = false;
  for (int i = 0; i < 300 && m_active.load() > 0; ++i) {
    usleep(100000);
  }
}

void HttpServer::accept_loop() {
  while (!m_stop_requested && m_listen_fd >= 0) {
    struct timeval tv;
    tv.tv_sec = 1;
    tv.tv_usec = 0;
    fd_set rf;
    FD_ZERO(&rf);
    FD_SET(m_listen_fd, &rf);
    int sr = select(m_listen_fd + 1, &rf, NULL, NULL, &tv);
    if (sr < 0) {
      if (errno == EINTR) {
        continue;
      }
      break;
    }
    if (sr == 0) {
      continue;
    }
    struct sockaddr_in pa;
    socklen_t pl = sizeof(pa);
    int cfd = accept(m_listen_fd, (struct sockaddr*)&pa, &pl);
    if (cfd < 0) {
      if (m_stop_requested) {
        break;
      }
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      break;
    }
    if (m_active.load() >= m_opt.max_connections) {
      const char* body = "{\"error\":{\"code\":503,\"message\":\"too many concurrent connections\","
                         "\"type\":\"server_overloaded\"}}";
      std::string r = "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: " +
                      std::to_string(strlen(body)) + "\r\nConnection: close\r\n\r\n";
      r += body;
      write(cfd, r.data(), r.size());
      close(cfd);
      continue;
    }
    char ip[64];
    inet_ntop(AF_INET, &pa.sin_addr, ip, sizeof(ip));
    std::string peer = std::string(ip) + ":" + std::to_string(pa.sin_port);
    m_active++;
    m_total++;
    std::thread th(&HttpServer::handle_conn, this, cfd, peer);
    th.detach();
  }
  m_running = false;
}

static std::string lower_str(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  return s;
}

void HttpServer::handle_conn(int fd, const std::string& peer) {
  set_sock_timeout(fd, m_opt.write_timeout_sec, m_opt.idle_keepalive_sec);
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

  bool keep_alive = true;
  int nreq = 0;
  std::string buf;  // 跨请求保留，支持 keep-alive / pipelining
  char tmp[8192];

  while (keep_alive && !m_stop_requested && nreq < m_opt.max_requests_per_conn) {
    while (buf.find("\r\n\r\n") == std::string::npos) {
      ssize_t n = read(fd, tmp, sizeof(tmp));
      if (n <= 0) {
        if (n < 0 && errno == EINTR) {
          continue;
        }
        goto done;  // 对端关闭 / 空闲超时
      }
      buf.append(tmp, (size_t)n);
      if (buf.size() > (size_t)m_opt.max_header_bytes) {
        SocketWriter w(fd, false);
        w.write_full(431, "{\"error\":{\"code\":431,\"message\":\"headers too large\"}}", "application/json");
        goto done;
      }
    }

    HttpRequest req;
    req.peer = peer;
    size_t hend = buf.find("\r\n\r\n");
    std::string head = buf.substr(0, hend);
    buf.erase(0, hend + 4);

    size_t nl = head.find("\r\n");
    std::string line = head.substr(0, nl);
    size_t sp1 = line.find(' ');
    size_t sp2 = line.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) {
      SocketWriter w(fd, false);
      w.write_full(400, "{\"error\":{\"code\":400,\"message\":\"malformed request line\"}}", "application/json");
      goto done;
    }
    req.method = line.substr(0, sp1);
    req.target = line.substr(sp1 + 1, sp2 - sp1 - 1);
    req.version = (sp2 + 1 < line.size()) ? line.substr(sp2 + 1) : "HTTP/1.0";
    size_t qm = req.target.find('?');
    if (qm == std::string::npos) {
      req.path = req.target;
    } else {
      req.path = req.target.substr(0, qm);
      req.query = req.target.substr(qm + 1);
    }

    bool want_close = false;
    size_t pos = nl + 2;
    while (pos < head.size()) {
      size_t e = head.find("\r\n", pos);
      if (e == std::string::npos) {
        e = head.size();
      }
      std::string h = head.substr(pos, e - pos);
      size_t colon = h.find(':');
      if (colon != std::string::npos) {
        std::string k = lower_str(h.substr(0, colon));
        std::string v = h.substr(colon + 1);
        size_t s = v.find_first_not_of(" \t");
        if (s != std::string::npos) {
          v = v.substr(s);
        }
        size_t t = v.find_last_not_of(" \t");
        if (t != std::string::npos) {
          v = v.substr(0, t + 1);
        }
        req.headers[k] = v;
      }
      pos = e + 2;
    }

    std::map<std::string, std::string>::iterator itc = req.headers.find("connection");
    if (itc != req.headers.end() && lower_str(itc->second).find("close") != std::string::npos) {
      want_close = true;
    }
    if (req.version != "HTTP/1.1") {
      want_close = true;
    }
    keep_alive = !want_close;
    nreq++;

    long clen = 0;
    std::map<std::string, std::string>::iterator it = req.headers.find("content-length");
    if (it != req.headers.end()) {
      clen = atol(it->second.c_str());
    }
    std::string te = req.headers.count("transfer-encoding") ? lower_str(req.headers["transfer-encoding"]) : "";
    if (te.find("chunked") != std::string::npos) {
      SocketWriter w(fd, false);
      w.write_full(501, "{\"error\":{\"code\":501,\"message\":\"chunked request bodies unsupported\"}}",
                   "application/json");
      goto done;
    }
    if (clen < 0 || clen > (long)m_opt.max_body_bytes) {
      SocketWriter w(fd, false);
      w.write_full(413, "{\"error\":{\"code\":413,\"message\":\"request body too large\"}}", "application/json");
      goto done;
    }
    if (req.headers.count("expect") && lower_str(req.headers["expect"]).find("100-continue") != std::string::npos) {
      static const char* cont = "HTTP/1.1 100 Continue\r\n\r\n";
      write(fd, cont, strlen(cont));
    }
    while ((long)buf.size() < clen) {
      ssize_t n = read(fd, tmp, sizeof(tmp));
      if (n <= 0) {
        if (n < 0 && errno == EINTR) {
          continue;
        }
        goto done;
      }
      buf.append(tmp, (size_t)n);
    }
    req.body = buf.substr(0, (size_t)clen);
    buf.erase(0, (size_t)clen);

    SocketWriter w(fd, keep_alive);
    try {
      m_handler(req, w);
    } catch (const std::exception& e) {
      std::string msg;
      for (size_t i = 0; i < strlen(e.what()); ++i) {
        char ch = e.what()[i];
        if (ch == '"' || ch == '\\') {
          msg += '\\';
          msg += ch;
        } else if ((unsigned char)ch < 0x20) {
          msg += ' ';
        } else {
          msg += ch;
        }
      }
      fprintf(stderr, "handler exception on %s %s: %s\n", req.method.c_str(), req.path.c_str(), e.what());
      fflush(stderr);
      if (w.alive()) {
        w.write_full(500, std::string("{\"error\":{\"code\":500,\"message\":\"") + msg +
                               "\",\"type\":\"server_error\"}}", "application/json");
      }
    }
    if (!w.keep_alive() || !w.alive()) {
      keep_alive = false;
    }
    // 空闲超时改为 keep-alive 语义
    set_sock_timeout(fd, m_opt.write_timeout_sec, m_opt.idle_keepalive_sec);
  }
done:
  close(fd);
  m_active--;
}
