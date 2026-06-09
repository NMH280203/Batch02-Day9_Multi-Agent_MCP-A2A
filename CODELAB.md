# Codelab: Xây Dựng Hệ Thống Multi-Agent với A2A Protocol

**Thời gian:** 2 giờ  
**Ngôn ngữ:** Python 3.11+  
**Công nghệ:** LangGraph, LangChain, A2A SDK

## Mục Tiêu Học Tập

Sau khi hoàn thành codelab này, bạn sẽ:
- Hiểu cách LLM hoạt động từ cơ bản đến nâng cao
- Biết cách tích hợp tools và RAG vào LLM
- Xây dựng được single agent với ReAct pattern
- Tạo multi-agent system với LangGraph
- Triển khai distributed agents với A2A protocol

## Chuẩn Bị

### Yêu Cầu Hệ Thống
- Python 3.11 trở lên
- [uv](https://docs.astral.sh/uv/) package manager
- API key từ [OpenAI](https://platform.openai.com) hoặc [OpenRouter](https://openrouter.ai)

### Cài Đặt

```bash
# Clone repository
git clone <repo-url>
cd legal_multiagent

# Cài đặt dependencies
uv sync

# Cấu hình environment
cp .env.example .env
# Sửa file .env, thêm OPENAI_API_KEY của bạn
```

---

## Phần 1: Direct LLM Calling (20 phút)

### Lý Thuyết

LLM (Large Language Model) ở dạng cơ bản nhất là một API nhận input text và trả về output text. Không có memory, không có tools, chỉ dựa vào training data.

**Ưu điểm:**
- Đơn giản, dễ implement
- Phản hồi nhanh

**Nhược điểm:**
- Không có kiến thức real-time
- Không thể tra cứu database
- Không có context giữa các lần gọi

### Thực Hành

**Bước 1:** Chạy demo Stage 1

```bash
uv run python stages/stage_1_direct_llm/main.py
```

**Bước 2:** Đọc và hiểu code

Mở file `stages/stage_1_direct_llm/main.py` và trả lời:

1. LLM được khởi tạo như thế nào? (Tìm hàm `get_llm()`)
2. Message được gửi đến LLM có cấu trúc gì?
3. Tại sao cần có `SystemMessage` và `HumanMessage`?

**Bài Tập 1.1:** Thay đổi câu hỏi

Sửa biến `QUESTION` thành câu hỏi pháp lý khác (tiếng Việt hoặc tiếng Anh) và chạy lại.

**Bài Tập 1.2:** Thêm temperature control

Thêm parameter `temperature=0.3` vào hàm `get_llm()` trong `common/llm.py` để làm output ổn định hơn.

### ✅ Đáp án Phần 1

**Bước 2 — Trả lời câu hỏi:**

1. **LLM được khởi tạo như thế nào?** Gọi `get_llm()` trong `common/llm.py`, trả về `ChatOpenAI` trỏ tới OpenAI API với model `gpt-4o-mini`, `temperature=0.3`, `max_tokens=1024`.
2. **Cấu trúc message?** List gồm `SystemMessage` (vai trò + hướng dẫn) và `HumanMessage` (câu hỏi người dùng), truyền vào `llm.ainvoke(messages)`.
3. **Tại sao cần SystemMessage và HumanMessage?** `SystemMessage` định nghĩa persona và ràng buộc output; `HumanMessage` chứa input thực tế. Tách biệt giúp LLM phân vai rõ ràng.

**Bài tập 1.1 — Đã làm:** Đổi `QUESTION` sang tiếng Việt về chấm dứt HĐLĐ. Chạy thành công (~7s), LLM trả lời theo Bộ luật Lao động VN 2019.

**Bài tập 1.2 — Đã làm:** Thêm `temperature=0.3` vào `common/llm.py`. Output ổn định hơn, ít biến động giữa các lần chạy.

---

## Phần 2: LLM + RAG & Tools (30 phút)

### Lý Thuyết

**RAG (Retrieval-Augmented Generation):** Cho phép LLM tra cứu knowledge base trước khi trả lời.

**Tools:** Các function mà LLM có thể gọi để thực hiện tác vụ cụ thể (tính toán, query database, gọi API).

**Function Calling Flow:**
1. LLM nhận câu hỏi + danh sách tools
2. LLM quyết định gọi tool nào (hoặc không gọi)
3. Tool được execute, trả về kết quả
4. LLM nhận kết quả và tạo câu trả lời cuối cùng

### Thực Hành

**Bước 1:** Chạy demo Stage 2

```bash
uv run python stages/stage_2_rag_tools/main.py
```

**Bước 2:** Phân tích code

Mở `stages/stage_2_rag_tools/main.py` và tìm:

1. Hàm `@tool` decorator được dùng ở đâu?
2. `LEGAL_KNOWLEDGE` được cấu trúc như thế nào?
3. LLM được bind với tools ra sao? (Tìm `.bind_tools()`)

**Bài Tập 2.1:** Thêm knowledge base entry

Thêm một entry mới vào `LEGAL_KNOWLEDGE` về luật lao động:

```python
{
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
    "text": (
        "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể "
        "đơn phương chấm dứt hợp đồng trong các trường hợp: (1) người lao động "
        "thường xuyên không hoàn thành công việc; (2) bị ốm đau, tai nạn đã điều trị "
        "12 tháng chưa khỏi; (3) thiên tai, hỏa hoạn; (4) người lao động đủ tuổi nghỉ hưu."
    ),
}
```

**Bài Tập 2.2:** Tạo tool mới

Tạo một tool `@tool` mới tên `check_statute_of_limitations` nhận vào `case_type` (string) và trả về thời hiệu khởi kiện:

```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.
    
    Args:
        case_type: Loại vụ án (contract, tort, property)
    """
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
    }
    return limits.get(case_type.lower(), "Không xác định")
```

Thêm tool này vào danh sách tools và test.

### ✅ Đáp án Phần 2

**Bước 2 — Trả lời câu hỏi:**

1. **`@tool` decorator ở đâu?** Trên `search_legal_database`, `calculate_damages`, và `check_statute_of_limitations` (dòng ~101–160).
2. **`LEGAL_KNOWLEDGE` cấu trúc?** List các dict, mỗi entry có `id`, `keywords` (list từ khóa), `text` (nội dung pháp lý). Search bằng keyword overlap.
3. **Bind tools?** `llm_with_tools = llm.bind_tools(TOOLS)` — LLM nhận schema tools và quyết định gọi tool nào.

**Bài tập 2.1 — Đã làm:** Thêm entry `labor_law` về Bộ luật Lao động VN 2019 vào `LEGAL_KNOWLEDGE`.

**Bài tập 2.2 — Đã làm:** Tạo `check_statute_of_limitations`, thêm vào `TOOLS`. Test: `contract` → `"4 năm (UCC § 2-725)"`. Chạy Stage 2 thành công (~9.5s), LLM gọi `search_legal_database` cho câu hỏi NDA.

---

## Phần 3: Single Agent với ReAct (25 phút)

### Lý Thuyết

**ReAct Pattern:** Reasoning + Acting

Agent tự động lặp lại chu trình:
1. **Think:** Suy nghĩ cần làm gì
2. **Act:** Gọi tool
3. **Observe:** Nhận kết quả
4. Lặp lại cho đến khi có câu trả lời cuối cùng

LangGraph cung cấp `create_react_agent` để tự động hóa pattern này.

### Thực Hành

**Bước 1:** Chạy demo Stage 3

```bash
uv run python stages/stage_3_single_agent/main.py
```

**Bước 2:** Quan sát output

Chú ý cách agent tự động:
- Quyết định tool nào cần gọi
- Gọi nhiều tools liên tiếp
- Tổng hợp kết quả

**Bước 3:** Đọc code

Mở `stages/stage_3_single_agent/main.py`:

1. Tìm `create_react_agent()` — đây là magic function
2. So sánh với Stage 2: không còn manual tool loop
3. Xem `agent_executor.invoke()` — chỉ cần gọi một lần

**Bài Tập 3.1:** Thêm tool tra cứu án lệ

```python
@tool
def search_case_law(keywords: str) -> str:
    """Tìm kiếm án lệ theo từ khóa.
    
    Args:
        keywords: Từ khóa tìm kiếm
    """
    cases = {
        "breach": "Hadley v. Baxendale (1854) - Consequential damages",
        "negligence": "Donoghue v. Stevenson (1932) - Duty of care",
        "contract": "Carlill v. Carbolic Smoke Ball Co (1893) - Unilateral contract",
    }
    for key, case in cases.items():
        if key in keywords.lower():
            return case
    return "Không tìm thấy án lệ phù hợp"
```

Thêm vào tools list và test với câu hỏi về breach of contract.

**Bài Tập 3.2:** Debug agent reasoning

Thêm `verbose=True` vào `create_react_agent()` để xem chi tiết quá trình suy nghĩ của agent.

### ✅ Đáp án Phần 3

**Bước 2 — Quan sát output (đã chạy):**
- Agent gọi `search_legal_database` + `search_case_law` song song (Step 1)
- Nhận kết quả từ tools (Step 2–3 OBSERVE)
- Tổng hợp câu trả lời cuối (Step 4 FINAL ANSWER)
- Tổng thời gian: ~14s

**Bước 3 — Trả lời câu hỏi:**

1. **`create_react_agent()`** — Tự động hóa vòng lặp Think→Act→Observe, không cần viết tool loop thủ công.
2. **So với Stage 2:** Stage 2 chỉ 1 vòng tool call (manual loop); Stage 3 agent tự quyết định gọi bao nhiêu tools, theo thứ tự nào.
3. **`invoke()` vs Stage 2:** Stage 3 chỉ cần `graph.astream(inputs)` một lần; agent tự orchestrate bên trong.

**Bài tập 3.1 — Đã làm:** Thêm `search_case_law`, đổi `QUESTION` về breach of contract. Agent gọi cả `search_legal_database` và `search_case_law`.

**Bài tập 3.2 — Đã làm:** `verbose=True` không còn hỗ trợ trong LangGraph v1.0+. Thay bằng `graph.astream(inputs, stream_mode="updates")` — in từng bước THINK+ACT / OBSERVE / FINAL ANSWER.

---

## Phần 4: Multi-Agent In-Process (30 phút)

### Lý Thuyết

**Multi-Agent System:** Nhiều agents chuyên môn hóa cùng làm việc.

**Ưu điểm:**
- Mỗi agent tập trung vào domain riêng
- Có thể chạy song song (parallel execution)
- Dễ maintain và mở rộng

**LangGraph StateGraph:**
- Định nghĩa state (dữ liệu chia sẻ giữa các nodes)
- Tạo nodes (các bước xử lý)
- Định nghĩa edges (luồng điều khiển)

**Send API:** Cho phép dispatch nhiều tasks song song.

### Thực Hành

**Bước 1:** Chạy demo Stage 4

```bash
uv run python stages/stage_4_milti_agent/main.py
```

**Bước 2:** Phân tích kiến trúc

Mở `stages/stage_4_milti_agent/main.py`:

1. Tìm `class State(TypedDict)` — đây là shared state
2. Tìm các agent functions: `law_agent`, `tax_agent`, `compliance_agent`
3. Tìm `Send()` API — dispatch parallel tasks
4. Xem `graph.add_node()` và `graph.add_edge()`

**Bước 3:** Vẽ graph

```python
# Thêm vào cuối file main.py
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

**Bài Tập 4.1:** Thêm agent mới

Tạo `privacy_agent` chuyên về GDPR và privacy law:

```python
def privacy_agent(state: State) -> dict:
    """Agent chuyên về luật bảo vệ dữ liệu cá nhân."""
    llm = get_llm()
    
    prompt = f"""Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân.
    
Câu hỏi gốc: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Hãy phân tích các vấn đề về privacy và GDPR (nếu có).
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}
```

Thêm node này vào graph và kết nối với `aggregate_results`.

**Bài Tập 4.2:** Implement conditional routing

Sửa `check_routing` để chỉ gọi privacy_agent khi câu hỏi có từ khóa "data", "privacy", "gdpr":

```python
def check_routing(state: State) -> list[Send]:
    question_lower = state["question"].lower()
    tasks = []
    
    if any(kw in question_lower for kw in ["tax", "irs", "thuế"]):
        tasks.append(Send("tax_agent", state))
    
    if any(kw in question_lower for kw in ["compliance", "sec", "regulation"]):
        tasks.append(Send("compliance_agent", state))
    
    if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu"]):
        tasks.append(Send("privacy_agent", state))
    
    return tasks if tasks else [Send("aggregate_results", state)]
```

### ✅ Đáp án Phần 4

**Bước 2 — Trả lời câu hỏi:**

1. **Shared state:** `class LegalState(TypedDict)` — chứa `question`, `law_analysis`, `tax_result`, `compliance_result`, `privacy_analysis`, `final_answer`.
2. **Agent functions:** `analyze_law`, `call_tax_specialist`, `call_compliance_specialist`, `privacy_agent`, `aggregate`.
3. **`Send()` API:** Trong `check_routing()` — dispatch parallel tasks tới specialist nodes.
4. **Graph:** `graph.add_node()` đăng ký nodes; `add_conditional_edges` + `add_edge` định nghĩa luồng.

**Bước 3 — Graph topology (Mermaid):**

```
analyze_law → check_routing ─┬→ call_tax_specialist ──────┐
                              ├→ call_compliance_specialist ┼→ aggregate → END
                              ├→ privacy_agent ─────────────┘
                              └→ aggregate (nếu không match keyword)
```

**Bài tập 4.1 — Đã làm:** Thêm `privacy_agent()` node, field `privacy_analysis` trong state, kết nối tới `aggregate`.

**Bài tập 4.2 — Đã làm:** Thay LLM routing bằng keyword routing trong `check_routing()`. Chạy thử: câu hỏi có "tax" → dispatch `call_tax_specialist` (~26s tổng).

---

## Phần 5: Distributed A2A System (15 phút)

### Lý Thuyết

**A2A (Agent-to-Agent) Protocol:** Chuẩn giao tiếp giữa các agents qua HTTP.

**Khác biệt với Stage 4:**
- Mỗi agent là một service độc lập
- Giao tiếp qua HTTP thay vì in-process
- Dynamic discovery qua Registry
- Có thể scale từng agent riêng biệt

**Kiến trúc:**
```
Registry (10000) ← agents register on startup
    ↓
Customer Agent (10100) → Law Agent (10101)
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
            Tax Agent (10102)   Compliance Agent (10103)
```

### Thực Hành

**Bước 1:** Khởi động toàn bộ hệ thống

```bash
./start_all.sh
```

Chờ ~10 giây để tất cả services khởi động.

**Bước 2:** Test hệ thống

```bash
uv run python test_client.py
```

**Bước 3:** Quan sát logs

Mở 5 terminal tabs và xem logs của từng service:
- Registry: port 10000
- Customer Agent: port 10100
- Law Agent: port 10101
- Tax Agent: port 10102
- Compliance Agent: port 10103

**Bài Tập 5.1:** Trace request flow

Trong logs, tìm `trace_id` và theo dõi request đi qua các agents. Vẽ sequence diagram.

**Bài Tập 5.2:** Test dynamic discovery

1. Dừng Tax Agent (Ctrl+C)
2. Chạy lại `test_client.py`
3. Quan sát lỗi và cách hệ thống xử lý

**Bài Tập 5.3:** Modify agent behavior

Sửa `tax_agent/graph.py`, thay đổi system prompt để agent trả lời ngắn gọn hơn. Restart tax agent và test lại.

### ✅ Đáp án Phần 5

**Bài tập 5.1 — Trace request flow:**

```
User → Customer Agent (sinh trace_id)
         → Registry discover("legal_question") → Law Agent
              → analyze_law
              → check_routing
              ├→ Registry discover → Tax Agent (depth=2, cùng trace_id)
              └→ Registry discover → Compliance Agent (depth=2, cùng trace_id)
              → aggregate → trả về Customer Agent → User
```

`trace_id` sinh tại Customer Agent (`customer_agent/agent_executor.py`), propagate qua metadata A2A message (`common/a2a_client.py`). Dùng `context_id` từ `test_client.py` output để correlate logs giữa 5 services.

**Bài tập 5.2 — Test dynamic discovery (đã chạy):**

1. Dừng Tax Agent: `kill $(lsof -ti :10102)`
2. Chạy `test_client.py` → hệ thống **vẫn trả lời** (~39s)
3. Law Agent bắt exception trong `call_tax()`, ghi: `[Tax analysis unavailable: ...]` rồi tiếp tục aggregate — **graceful degradation**, không crash toàn hệ thống.

**Bài tập 5.3 — Đã làm:** Rút gọn `TAX_SYSTEM_PROMPT` trong `tax_agent/graph.py` — yêu cầu trả lời dưới 150 từ, dùng bullet points.

---

## Phần 6: Tổng Kết & Mở Rộng (10 phút)

### So Sánh 5 Stages

| Stage | Pattern | Use Case | Complexity |
|---|---|---|---|
| 1 | Direct LLM | Câu hỏi đơn giản, không cần tools | ⭐ |
| 2 | LLM + Tools | Cần tra cứu data hoặc tính toán | ⭐⭐ |
| 3 | ReAct Agent | Tự động orchestration, multi-step | ⭐⭐⭐ |
| 4 | Multi-Agent | Nhiều domains, parallel processing | ⭐⭐⭐⭐ |
| 5 | Distributed A2A | Production, scalable, fault-tolerant | ⭐⭐⭐⭐⭐ |

### Câu Hỏi Ôn Tập

1. Khi nào nên dùng single agent thay vì multi-agent?
2. Ưu điểm của A2A protocol so với gRPC hoặc REST thông thường?
3. Làm thế nào để prevent infinite delegation loops trong A2A?
4. Tại sao cần Registry service? Có thể hardcode URLs không?

### ✅ Đáp án Câu Hỏi Ôn Tập

1. **Single agent khi nào?** Câu hỏi đơn domain, ít bước, không cần parallelism — ví dụ tra cứu pháp lý đơn giản (Stage 3 đủ). Multi-agent khi cần chuyên môn hóa domain + chạy song song.
2. **A2A vs gRPC/REST thường:** A2A có Agent Card chuẩn (capabilities, skills), task lifecycle, trace propagation — thiết kế riêng cho agent-to-agent, không cần tự định nghĩa protocol.
3. **Prevent infinite loops:** `MAX_DELEGATION_DEPTH = 3` trong `law_agent/graph.py` — khi `depth >= 3`, bỏ qua sub-agent delegation.
4. **Registry:** Dynamic discovery, agents tự register/deregister. Hardcode URLs được cho demo nhỏ, nhưng production cần Registry để scale, failover, và thay đổi endpoint không sửa code.

### Bài Tập Nâng Cao (Tự Học)

**Challenge 1:** Thêm memory/conversation history

Implement conversation memory để agent nhớ các câu hỏi trước đó.

**Challenge 2:** Add authentication

Thêm API key authentication cho các A2A endpoints.

**Challenge 3:** Implement retry logic

Khi một agent fail, tự động retry với exponential backoff.

**Challenge 4:** Monitoring & Observability

Tích hợp LangSmith hoặc Prometheus để monitor agent performance.

---

## Tài Liệu Tham Khảo

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [A2A Protocol Spec](https://github.com/google/A2A)
- [OpenRouter API](https://openrouter.ai/docs)
- Architecture diagrams: `docs/*.svg`

## Hỗ Trợ

Nếu gặp vấn đề:
1. Check `.env` file có đúng API key không
2. Đảm bảo tất cả ports (10000-10103) không bị chiếm
3. Xem logs trong terminal để debug
4. Đọc error messages cẩn thận — thường có hint rõ ràng

---

## **Bài Tập Cộng Điểm:**

Sau khi chạy full Stage 5 (`test_client.py`) trả lời 2 câu hỏi:

### ✅ Đáp án Bài Tập Cộng Điểm

**1. Latency baseline:**

| Metric | Giá trị |
|---|---|
| Model | `gpt-4o-mini` |
| `max_tokens` | 1024 |
| **Tổng latency** | **71.81 giây** |
| Số LLM hops | ~5–7 (Customer → Law → Tax + Compliance song song → Aggregate) |

**2. Phương án giảm latency + demo:**

| Phương án | Mô tả | Kết quả đo |
|---|---|---|
| Giảm `max_tokens` | `OPENAI_MAX_TOKENS=256` thay vì 1024 — ít token generate hơn mỗi hop | **35.36 giây** (−51%) |
| Model nhanh hơn | Dùng `gpt-4o-mini` thay model lớn | Đã áp dụng |
| Parallel delegation | Tax + Compliance chạy song song qua `Send` API | Đã có sẵn trong Law Agent |

**Cách reproduce:**
```bash
# Baseline (max_tokens=1024)
./start_all.sh
uv run python test_client.py
# → [LATENCY] Total response time: 71.81s

# Optimized (max_tokens=256)
kill $(lsof -ti :10000,:10100,:10101,:10102,:10103)
OPENAI_MAX_TOKENS=256 ./start_all.sh
uv run python test_client.py
# → [LATENCY] Total response time: 35.36s
```

**Kết luận:** Latency chủ yếu do **chuỗi nhiều lần gọi LLM tuần tự** (Customer → Law → Aggregate). Giảm `max_tokens` là cách nhanh nhất để cắt ~50% thời gian mà không đổi kiến trúc.

**Chúc các bạn học tốt! 🚀**
