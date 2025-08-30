.PHONY: setup start stop clean logs

LOG_DIR := .logs
PID_DIR := .pids

# === 全新启动（首次拉取，安装所有依赖 + 启动服务） ===
setup:
	@echo "📦 安装后端依赖..."
	cd backend && uv sync
	@echo "📦 安装前端依赖..."
	cd frontend && npm install
	@$(MAKE) start

# === 日常启动（依赖已安装，直接启动） ===
start:
	@mkdir -p $(LOG_DIR) $(PID_DIR)
	@echo "🐳 启动基础设施（等待健康检查通过，Milvus 冷启动可能需要 1-2 分钟）..."
	cd backend && docker compose up -d --wait
	@echo "🚀 启动后端 API..."
	@cd backend && nohup uv run python scripts/server.py --reload > ../$(LOG_DIR)/backend.log 2>&1 & echo $$! > $(PID_DIR)/backend.pid
	@echo "🚀 启动前端..."
	@cd frontend && nohup npm run dev > ../$(LOG_DIR)/frontend.log 2>&1 & echo $$! > $(PID_DIR)/frontend.pid
	@echo ""
	@echo "✅ 启动完成！"
	@echo "   前端: http://localhost:5173"
	@echo "   后端: http://localhost:8000"
	@echo "   日志: make logs    停止: make stop"

# === 查看前后端日志 ===
logs:
	tail -f $(LOG_DIR)/backend.log $(LOG_DIR)/frontend.log

# === 停止所有服务 ===
stop:
	@echo "停止前端和后端进程..."
	-@kill $$(cat $(PID_DIR)/backend.pid 2>/dev/null) 2>/dev/null || true
	-@kill $$(cat $(PID_DIR)/frontend.pid 2>/dev/null) 2>/dev/null || true
	-@pkill -f "scripts/server.py" 2>/dev/null || true
	-@pkill -f "vite" 2>/dev/null || true
	@rm -f $(PID_DIR)/backend.pid $(PID_DIR)/frontend.pid
	@echo "停止 Docker 服务..."
	cd backend && docker compose down
	@echo "✅ 已停止所有服务"

# === 清理（删除依赖、Docker 数据） ===
clean: stop
	rm -rf backend/.venv
	rm -rf frontend/node_modules frontend/dist
	rm -rf $(LOG_DIR) $(PID_DIR)
	cd backend && docker compose down -v
	@echo "✅ 已清理所有依赖和数据"
