// SPDX-FileCopyrightText: 2026 yuunnn-w
// SPDX-License-Identifier: GPL-3.0-only
// 本文件属 RK1828-Spark-X2.6-4B 项目；许可证见 LICENSE（GPL-3.0）与 NOTICE（第三方部件）。
// endpoints.h — OpenAI 端点与运维路由
#ifndef _RKNN_SPARK_ENDPOINTS_H_
#define _RKNN_SPARK_ENDPOINTS_H_

#include "app.h"

void dispatch(const HttpRequest& rq, ResponseWriter& w);

#endif
