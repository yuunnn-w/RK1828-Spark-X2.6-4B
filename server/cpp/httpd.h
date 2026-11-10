// SPDX-FileCopyrightText: 2026 yuunnn-w
// SPDX-License-Identifier: GPL-3.0-only
// 本文件属 RK1828-Spark-X2.6-4B 项目；许可证见 LICENSE（GPL-3.0）与 NOTICE（第三方部件）。
// httpd.h — 最小但生产可用的 HTTP/1.1 服务层（keep-alive、超时、连接上限、SSE）
#ifndef _RKNN_SPARK_HTTPD_H_
#define _RKNN_SPARK_HTTPD_H_

#include <stdint.h>

#include <atomic>
#include <functional>
#include <map>
#include <string>
#include <vector>

struct HttpRequest {
  std::string method;
  std::string path;      // 不含 query
  std::string target;    // 原始 request-target
  std::string query;
  std::string version;   // HTTP/1.1
  std::string body;
  std::string peer;      // ip:port
  std::map<std::string, std::string> headers;  // key 已小写
  long t_recv_us = 0;
};

class ResponseWriter {
 public:
  virtual ~ResponseWriter() {}
  virtual void write_full(int status, const std::string& body, const std::string& ctype) = 0;
  virtual void write_status_only(int status, const std::string& ctype) = 0;  // HEAD
  virtual bool write_sse_start() = 0;
  virtual bool write_sse_data(const std::string& payload) = 0;
  virtual void write_sse_end() = 0;
  virtual bool alive() = 0;
};

typedef std::function<void(const HttpRequest&, ResponseWriter&)> HttpHandler;

struct HttpServerOptions {
  std::string host = "0.0.0.0";
  int port = 8080;
  int backlog = 128;
  int max_connections = 128;     // 超过直接 503
  int max_body_bytes = 32 * 1024 * 1024;
  int max_header_bytes = 64 * 1024;
  int read_timeout_sec = 30;     // 读请求头/体的空闲超时
  int write_timeout_sec = 900;   // 写超时
  int idle_keepalive_sec = 45;   // keep-alive 空闲超时
  int max_requests_per_conn = 2000;
  std::string server_name = "rknn3-spark-server/1.0";
  std::string api_key;           // 非空则要求 Authorization: Bearer <key>
};

class HttpServer {
 public:
  HttpServer();
  ~HttpServer();

  int start(const HttpServerOptions& opt, HttpHandler handler);
  void stop();          // 停止 accept，等待活跃连接结束
  bool running() const { return m_running; }
  int active_conns() const { return m_active; }
  long total_conns() const { return m_total; }

 private:
  void accept_loop();
  void handle_conn(int fd, const std::string& peer);

  HttpServerOptions m_opt;
  HttpHandler m_handler;
  int m_listen_fd = -1;
  std::atomic<bool> m_running;
  std::atomic<bool> m_stop_requested;
  std::atomic<int> m_active;
  std::atomic<long> m_total;
};

#endif
