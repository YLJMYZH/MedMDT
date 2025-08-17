.PHONY: setup start stop clean

# === 全新启动（首次拉取，安装所有依赖 + 启动服务） ===
setup:
	@echo "📦 安装后端依赖..."
	cd backend && uv sync
	@echo "📦 安装前端依赖..."
	cd frontend && npm install
	@echo "🐳 启动基础设施（Neo4j / Milvus / Elasticsearch）..."
	cd backend && docker compose up -d
	@echo "⏳ 等待服务就绪..."
	sleep 10
	@echo "🚀 启动后端 API..."
	cd backend && uv run python scripts/server.py --reload &
	@echo "🚀 启动前端..."
	cd frontend && npm run dev &
	@echo ""
	@echo "✅ 全部启动完成！"
	@echo "   前端: http://localhost:5173"
	@echo "   后端: http://localhost:8000"

# === 日常启动（依赖已安装，直接启动） ===
start:
	@echo "🐳 启动基础设施..."
	cd backend && docker compose up -d
	@echo "🚀 启动后端 API..."
	cd backend && uv run python scripts/server.py --reload &
	@echo "🚀 启动前端..."
	cd frontend && npm run dev &
	@echo ""
	@echo "✅ 启动完成！"
	@echo "   前端: http://localhost:5173"
	@echo "   后端: http://localhost:8000"

# === 停止所有服务 ===
stop:
	@echo "停止前端和后端进程..."
	-pkill -f "vite" 2>/dev/null || true
	-pkill -f "uvicorn" 2>/dev/null || true
	@echo "停止 Docker 服务..."
	cd backend && docker compose down
	@echo "✅ 已停止所有服务"

# === 清理（删除依赖、Docker 数据） ===
clean: stop
	rm -rf backend/.venv
	rm -rf frontend/node_modules frontend/dist
	cd backend && docker compose down -v
	@echo "✅ 已清理所有依赖和数据"
