"""正则解析单条指令文本 → Case 对象"""
import re

from src.models.case import Branch, CallFlowStep, Case, Constraint, KnowledgePoint


# ============================================================
# 业务线识别
# ============================================================

BUSINESS_KEYWORDS = [
    ("外卖", ["外卖", "骑手", "闪购", "配送", "跑腿", "商家.*运营", "新店", "站长", "骑手助手"]),
    ("酒店", ["酒店", "民宿", "住宿", "预订", "入住", "满房", "房东"]),
    ("到店", ["到店", "团购", "券", "火锅", "套餐"]),
    ("打车", ["打车", "司机", "出车", "冲单", "租赁"]),
    ("医美", ["医美", "光子嫩肤", "术后", "医疗"]),
    ("企业订餐", ["企业订餐", "企业.*账户", "企客"]),
    ("充电宝", ["充电宝"]),
    ("门票", ["门票", "景点", "景区"]),
    ("招聘", ["招聘", "招募", "骑手.*合同", "飞毛腿", "骑手招聘"]),
    ("课程平台", ["课程", "直播", "培训机构", "校区"]),
    ("金融", ["金融", "还款", "贷款"]),
    ("商家服务", ["商家.*支付", "商家.*推广", "商家.*结算", "商家.*房源", "团长"]),
]


def detect_business_line(instruction: str) -> str:
    """根据指令文本自动识别业务线"""
    for line_name, keywords in BUSINESS_KEYWORDS:
        for kw in keywords:
            if re.search(kw, instruction):
                return line_name
    return "其他"


# ============================================================
# 区块提取
# ============================================================

def extract_section(text: str, *headings: str) -> str:
    """提取标题下的内容，直到下一个 # 标题为止"""
    heading = headings[0]
    for h in headings:
        if h in text:
            heading = h
            break

    idx = text.find(heading)
    if idx == -1:
        return ""

    content_start = idx + len(heading)
    content = text[content_start:]

    # 截到下一个一级标题（排除 ## 子标题）
    next_heading = re.search(r"\n(?=#[^#])|\n(?=\n#[^#])|\Z", content)
    if next_heading:
        content = content[:next_heading.start()]

    return content.strip()


# ============================================================
# 各部分解析
# ============================================================

def parse_role(instruction: str) -> str:
    """提取角色"""
    for heading in ["# Role:", "# Role\n", "# Role\r\n"]:
        if heading in instruction:
            content = extract_section(instruction, heading)
            return content.split("\n")[0].strip()
    return ""


def parse_task(instruction: str) -> str:
    """提取任务描述"""
    content = extract_section(instruction, "# Task\n", "# Task:", "## Task:", "## Task\n")
    return content.strip()


def parse_opening_line(instruction: str) -> str:
    """提取开场白"""
    content = extract_section(instruction, "# Opening Line:", "# Opening Line\n")
    if not content:
        return ""
    # 取第一句有意义的
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    for line in lines:
        if "您好" in line or "你好" in line or "请问" in line:
            return line
    return lines[0] if lines else ""


def parse_call_flow(instruction: str) -> list[CallFlowStep]:
    """提取通话流程"""
    # 尝试两种格式
    content = extract_section(
        instruction,
        "# Call Flow\n", "# Conversation Flow:", "# Conversation Flow\n",
        "# Call Flow:", "## Call Flow\n",
    )
    if not content:
        return []

    # 格式B: ## Step N: 标题
    step_pattern_b = re.findall(
        r"##\s*Step\s*(\d+)\s*:?\s*(.*?)(?=\n##\s*Step|\n#|\Z)",
        content, re.DOTALL
    )
    if step_pattern_b:
        steps = []
        for step_num_str, step_body in step_pattern_b:
            step_num = int(step_num_str)
            lines = step_body.strip().split("\n")
            title = lines[0].strip() if lines else f"步骤{step_num}"

            branching = _extract_branches(step_body)
            sub_steps = [l.strip("- ").strip() for l in lines[1:]
                         if l.strip().startswith("- ") and "→" not in l]
            ref_script = ""
            ref_match = re.search(r"\*\*参考话术[：:]\*\*\s*(.*?)(?=\n\n|\n###|\Z)", step_body, re.DOTALL)
            if ref_match:
                ref_script = ref_match.group(1).strip()

            steps.append(CallFlowStep(
                id=f"step_{step_num}",
                step_number=step_num,
                title=title,
                description="",
                branching=branching,
                sub_steps=sub_steps,
                reference_script=ref_script,
                is_optional=False,
            ))
        return steps

    # 格式A: 数字序号
    step_blocks = re.split(r"\n(?=\d+\.\s)", content)
    if len(step_blocks) <= 1:
        return []

    steps = []
    for i, block in enumerate(step_blocks):
        block = block.strip()
        if not block:
            continue
        m = re.match(r"(\d+)\.\s*(.*)", block)
        if not m:
            continue
        step_num = int(m.group(1))
        first_line = m.group(2)
        rest = block[len(m.group(0)):].strip()

        title = first_line.split("。")[0].split("：")[0].split("，")[0].strip()
        if len(title) > 30:
            title = title[:30]

        branching = _extract_branches(rest)
        sub_steps = [l.strip("- ").strip() for l in rest.split("\n")
                     if l.strip().startswith("- ") and "→" not in l]

        steps.append(CallFlowStep(
            id=f"step_{step_num}",
            step_number=step_num,
            title=title,
            description=first_line,
            branching=branching,
            sub_steps=sub_steps,
            reference_script="",
            is_optional=False,
        ))
    return steps


def _extract_branches(text: str) -> list[Branch]:
    """从文本中提取分支条件"""
    branches = []
    # 模式:  - 条件 → 动作 / 进入第N步
    pattern = re.findall(r"-\s*(.+?)\s*[→>]\s*(.+)", text)
    for condition, action in pattern:
        cond = condition.strip()
        act = action.strip()
        # 尝试提取目标步骤
        target = 0
        target_match = re.search(r"(\d+)", act)
        if target_match:
            target = int(target_match.group(1))
        branches.append(Branch(condition=cond, target_step=target, action=act))
    return branches


def parse_knowledge_points(instruction: str) -> list[KnowledgePoint]:
    """提取FAQ知识点"""
    content = extract_section(instruction, "# Knowledge Points", "# Knowledge Points (FAQ)",
                              "# Knowledge Points:", "# FAQ", "## FAQ")
    if not content:
        return []

    items = re.findall(r"[-•]\s*(.+?)(?=\n[-•]|\Z)", content, re.DOTALL)
    kps = []
    for i, item in enumerate(items):
        item = item.strip()
        # 优先用冒号分割，其次用第一个句号（但不能在末尾）
        sep = re.search(r"[：:]", item)
        if not sep:
            sep = re.search(r"。", item)
            if sep and sep.start() == len(item) - 1:
                sep = None  # 句号在末尾，不做分割
        if sep:
            topic = item[:sep.start()].strip()
            cont = item[sep.end():].strip() or item  # 若分割后为空，保留全文
        else:
            topic = item[:20]
            cont = item
        kps.append(KnowledgePoint(id=f"kp{i+1}", topic=topic, content=cont))
    return kps


def parse_constraints(instruction: str) -> list[Constraint]:
    """提取约束条件并自动分类"""
    content = extract_section(instruction, "# Constraints:", "# Constraints\n",
                              "## Constraints:", "## Constraints\n")
    if not content:
        return []

    items = re.findall(r"[-•]\s*(.+?)(?=\n[-•]|\n\n|\Z)", content, re.DOTALL)
    constraints = []
    for i, item in enumerate(items):
        item = item.strip().replace("\n", " ")
        c_type, checkable, pattern = _classify_constraint(item)
        constraints.append(Constraint(
            id=f"c{i+1}",
            type=c_type,
            description=item,
            checkable_by_rule=checkable,
            rule_pattern=pattern,
        ))
    return constraints


def _classify_constraint(text: str) -> tuple[str, bool, str | None]:
    """自动分类一条约束"""
    # 字数限制
    if re.search(r"字(数|以内)?", text):
        m = re.search(r"(\d+)", text)
        if m:
            return ("word_limit", True, rf"^.{{1,{m.group(1)}}}$")
        return ("word_limit", True, None)

    # 禁用词/禁止表达
    if re.search(r"不说|禁止|禁用|不要[说提]", text):
        words = re.findall(r"[「「](.*?)[」」]|\"(.*?)\"", text)
        flat = [w for pair in words for w in pair if w]
        if flat:
            return ("forbidden_word", True, "|".join(re.escape(w) for w in flat))
        return ("forbidden_word", True, None)

    # 竞品提及
    if re.search(r"不.*(对比|提及|评价).*(平台|滴滴|京东|顺丰|饿了么)", text):
        names = re.findall(r"(滴滴|京东|顺丰|饿了么|抖音)", text)
        if names:
            return ("forbidden_word", True, "|".join(names))

    # 语气/态度
    if re.search(r"语气|态度|自然|亲切|随意|温柔|关怀", text):
        return ("tone", False, None)

    # 安全/验证
    if re.search(r"安全|验证|核实.*身份|密码|冻结", text):
        return ("safety", False, None)

    # 行为规范
    if re.search(r"不[得应能可要会主许准让必]\w*|必须|务必|坚持", text):
        return ("behavior", False, None)

    return ("other", False, None)


# ============================================================
# 主解析入口
# ============================================================

def parse_instruction(instruction: str, case_id: int, title: str = "") -> Case:
    """将一条完整指令文本解析为 Case 对象"""
    role = parse_role(instruction)
    task = parse_task(instruction)
    opening = parse_opening_line(instruction)
    call_flow = parse_call_flow(instruction)
    knowledge = parse_knowledge_points(instruction)
    constraints = parse_constraints(instruction)
    biz = detect_business_line(instruction)

    if not title:
        if task:
            title = task.split("。")[0].split("，")[0][:50]
        else:
            title = f"Case #{case_id}"

    return Case(
        id=case_id,
        title=title,
        business_line=biz,
        role=role,
        task=task,
        opening_line=opening,
        call_flow=call_flow,
        knowledge_points=knowledge,
        constraints=constraints,
        complexity_score=0.0,
        raw_instruction=instruction,
    )
