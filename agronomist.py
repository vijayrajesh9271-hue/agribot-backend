import os
import json
import asyncio
from dotenv import load_dotenv
import ee  # Google Earth Engine
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
from guards import check_for_abuse, handle_abuse_response
from tavily import TavilyClient
from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_export_api_key,
    ModelSettings,
    RunContextWrapper,
    function_tool,
    StopAtTools,
    FunctionTool,
    handoff
)
from agents.extensions.handoff_filters import remove_all_tools

_context_cache = {}

from db import (
    init_db,
    async_save_message,
    async_get_history,
    async_get_or_create_user,
    async_get_messages_for_conversation,  
)

load_dotenv() 
project_id = os.getenv("GEE_PROJECT_ID")
if project_id:
    try:
        ee.Initialize(project=project_id)
        print(f"✅ Google Earth Engine initialized with project: {project_id}")
    except Exception as e:
        print(f"❌ GEE initialization failed: {e}")
else:
    print("⚠️ GEE_PROJECT_ID not set — NDVI crop monitoring will not work")

#free translate to urdu for rag
def translate_to_urdu_free(text: str) -> str:
    """Translate text to Urdu using Google Translate (free, no API key needed)"""
    try:
        from deep_translator import GoogleTranslator
        # Split into chunks of 4500 chars (Google limit is 5000)
        chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
        translated_chunks = []
        for chunk in chunks:
            translated = GoogleTranslator(source='auto', target='ur').translate(chunk)
            translated_chunks.append(translated)
        return "\n".join(translated_chunks)
    except Exception as e:
        print(f"⚠️ Free translation failed: {e} — returning original")
        return text


# Model and external clients

class Context:
    def __init__(self):
        load_dotenv()
        tracing_api_key = os.getenv("OPENAI_API_KEY")
        if tracing_api_key:
            set_tracing_export_api_key(tracing_api_key)

        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")

        # Gemini-compatible OpenAI interface
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

        self.tavily = TavilyClient(api_key=self.tavily_api_key)
        self.external_client = AsyncOpenAI(api_key=self.gemini_api_key, base_url=self.base_url)

        self.model = OpenAIChatCompletionsModel(
            model="gemini-2.5-flash",
            openai_client=self.external_client,
        )

ctx = Context()

# Helpers: retry + guaranteed vision-diagnosis formatting

async def _run_with_retry(agent, input_data, max_turns=5, max_retries=2, base_delay=2.0):
    """
    Runs the agent, retrying on transient Gemini overload (503 UNAVAILABLE).
    Does NOT retry on quota exhaustion (429) — that won't resolve within seconds,
    so it fails fast and lets process_with_agronomy_team's fallback handle it.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return await Runner.run(agent, input_data, max_turns=max_turns)
        except Exception as e:
            error_str = str(e)
            last_error = e
            is_transient = "UNAVAILABLE" in error_str or "503" in error_str or "high demand" in error_str.lower()
            if is_transient and attempt < max_retries:
                delay = base_delay * (attempt + 1)
                print(f"⏳ Gemini overloaded (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay}s...")
                await asyncio.sleep(delay)
                continue
            raise last_error
    raise last_error


def _format_vision_header(vision_data: Optional[Dict[str, Any]]) -> str:
    """
    Builds a small, consistently-formatted diagnosis line (crop + condition +
    confidence) for confirmed single-crop vision results. Prepended to the final
    response so the diagnosis always reaches the UI clearly, regardless of how
    the model phrases its own reply. Returns "" for uncertain/rejected/dual-crop
    modes, since there's no confirmed single diagnosis to show yet in those cases.
    """
    if not vision_data or vision_data.get("mode") not in ("auto", "manual"):
        return ""

    crop = vision_data.get("crop")
    label = vision_data.get("label")
    if not crop or not label:
        return ""

    confidence = vision_data.get("confidence", 0)
    status = vision_data.get("status", "")
    icon = "✅" if status == "Healthy" else "🌾"

    return f"{icon} **{crop} — {label}**\n_Confidence: {confidence:.0%}_\n\n"


# Tools

@function_tool
async def agri_search(query: str) -> str:
    """Search for agronomy-related information"""
    try:
        results = ctx.tavily.search(query=f"agronomy {query}", max_results=5)
    except Exception as e:
        return f"Search failed: {e}"

    if not results or not results.get("results"):
        return "No relevant results found."

    output = "\n🔍 Top agronomy results:\n"
    for idx, r in enumerate(results["results"], 1):
        output += f"{idx}. {r.get('title','No title')} — {r.get('url','')}\n"
    return output

class RegionInput(BaseModel):
    coordinates: List[List[List[float]]]

@function_tool
async def crop_monitoring(region: RegionInput, start_date: str, end_date: str, index: str = "NDVI") -> str:
    """Monitor crop health using satellite imagery"""
    try:
        polygon = ee.Geometry.Polygon(region.coordinates)
        collection = (ee.ImageCollection("COPERNICUS/S2")
                      .filterBounds(polygon)
                      .filterDate(start_date, end_date)
                      .map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI')))
        ndvi_stats = collection.mean().reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=polygon,
            scale=10
        ).getInfo()
        mean_ndvi = ndvi_stats.get("NDVI", None)
        return f"📊 Mean NDVI for {start_date} to {end_date}: {mean_ndvi:.3f}"
    except Exception as e:
        return f"Crop monitoring failed: {e}"

class AnalysisCounter(FunctionTool):
    def __init__(self):
        self._count = 0
        super().__init__(
            name="analysis_counter",
            description="Counts each crop monitoring or research analysis done.",
            params_json_schema={"type": "object", "properties": {}},
            on_invoke_tool=self.on_invoke_tool
        )

    async def on_invoke_tool(self, tool_context, arguments) -> str:
        self._count += 1
        return f"📈 Total analyses performed in this session: {self._count}"

counter_tool = AnalysisCounter()


# conversation context
async def get_conversation_context(conversation_id: int, max_messages: int = 20) -> str:
    """Get recent conversation context with caching for performance"""
    if not conversation_id:
        return ""
    
    # Check cache (valid for 5 seconds)
    cache_key = f"{conversation_id}_{max_messages}"
    if cache_key in _context_cache:
        cached_data, timestamp = _context_cache[cache_key]
        if (datetime.utcnow() - timestamp).seconds < 5:
            return cached_data
    
    messages = await async_get_messages_for_conversation(conversation_id)
    if not messages:
        return ""
    
    # Limit to recent messages
    recent = messages[-max_messages:] if len(messages) > max_messages else messages
    
    context_lines = []
    for msg in recent:
        role = "You" if msg.role == "assistant" else "User"
        # Limit each message to 500 chars
        content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
        context_lines.append(f"{role}: {content}")
    
    result = "\n".join(context_lines)
    
    # Cache the result
    _context_cache[cache_key] = (result, datetime.utcnow())
    
    return result


# Escalation

class EscalationData(BaseModel):
    reason: str
    field_id: str

async def on_escalation(ctx: RunContextWrapper, input_data: EscalationData):
    quick_response = f"🚨 Urgent issue detected in field {input_data.field_id}: {input_data.reason}.\n\n"
    quick_response += "✅ Quick advice:\n"
    quick_response += "- Inspect the crop immediately to confirm if pests (like fall armyworm or stem borers) are present.\n"
    quick_response += "- If leaves are yellowing, apply a quick foliar nitrogen spray (e.g., urea 2% solution).\n"
    quick_response += "- Remove and destroy heavily infested leaves if possible.\n"
    quick_response += "- If pest pressure is high, apply a recommended pesticide such as Spinosad or Emamectin Benzoate (follow local guidelines).\n"
    quick_response += "- Ensure adequate irrigation, as stress worsens yellowing.\n"
    quick_response += "\n👉 Would you like to escalate this to a professional agronomist for tailored guidance? (yes/no)"
    return quick_response

escalation_agent = Agent(
    name="AgriEscalationAgent",
    instructions="""
    Handle urgent soil or crop issues (e.g., pests, diseases, crop stress).

    - Always suggest a quick and practical immediate solution first.
    - Then, ask the user: "Would you like to escalate this to a professional agronomist for detailed support?"
    - If the user says YES → provide the following contact:
        Agronomist: vijy
        Phone: +92 123456789
    - If the user says NO → continue giving advice yourself.
    """
)

escalation_handoff = handoff(
    agent=escalation_agent,
    tool_name_override="escalate_to_expert",
    tool_description_override="Escalate complex agronomy issues to a specialist.",
    on_handoff=on_escalation,
    input_type=EscalationData,
    input_filter=remove_all_tools
)


# Agent factory

def create_base_agent(user_email: str, global_memory: str, conversation_context: str, vision_context: str = "", language_instructions=""):
    """Create agent with full memory and conversation context"""
    
    instructions = f"""You are an expert AI Agronomist assistant helping {user_email.split('@')[0]}.

IMPORTANT - USER CONTEXT:
{global_memory}

RECENT CONVERSATION (refer to this for context):
{conversation_context}

{vision_context if vision_context else ""}

{language_instructions}

Your responsibilities:
- Reference past conversations naturally (e.g., "As we discussed before...")
- Remember user's crops, location, and farming methods
- Use agri_search for research
- Use crop_monitoring for NDVI analysis
- Be conversational and remember what the user has told you

When the user refers to "last time" or "before", check the conversation context above."""

    return Agent(
        name=f"AgroDeepSearchAgent_{user_email}",
        instructions=instructions,
        tools=[agri_search, crop_monitoring],
        model=ctx.model,
        model_settings=ModelSettings(temperature=0.4),
        tool_use_behavior=StopAtTools(stop_at_tool_names=["crop_monitoring"]),
        handoffs=[escalation_handoff]
    )

soil_agent = Agent(
    name="SoilExpert",
    instructions="You specialize in soil health and fertility. Provide actionable advice.",
    tools=[agri_search, crop_monitoring],
    model=ctx.model,
    model_settings=ModelSettings(temperature=0.2),
    handoffs=[escalation_handoff]
)

crop_agent = Agent(
    name="CropExpert",
    instructions="""
    You are an expert agronomist specializing in crop yield estimation,
    crop rotation planning, and NDVI monitoring.

    - Always check for location (coordinates) and date range.
    - If missing, ask the user before making predictions.
    - If NDVI or other data is available, analyze it and give insights on crop health, stress, and growth stage.
    - Explain what additional info is required if data is insufficient.
    - Give context-aware guidance for yield, irrigation, fertilizer.
    """,
    tools=[agri_search, crop_monitoring],
    model=ctx.model,
    model_settings=ModelSettings(temperature=0.3),
    handoffs=[escalation_handoff]
)

economics_agent = Agent(
    name="AgriEconomist",
    instructions="Focus on agricultural economics, market trends, and profitability.",
    tools=[agri_search, crop_monitoring],
    model=ctx.model,
    model_settings=ModelSettings(temperature=0.2),
    handoffs=[escalation_handoff]
)


# Router

async def route_question_with_context(question: str, base_agent: Agent, context_agents: dict) -> List[Agent]:
    """Route question to appropriate experts with memory context.
    NOTE: not currently called by run_agronomy_team (which does its own inline
    routing below) — kept here in case other code paths use it."""

    router_prompt = f"""Which experts should answer: "{question}"
    Options: SoilExpert, CropExpert, AgriEconomist, BaseAgent
    Return comma-separated list."""
    
    router_result = await _run_with_retry(base_agent, router_prompt, max_turns=1)
    decision = router_result.final_output.lower()

    selected: List[Agent] = []
    if "soil" in decision:
        selected.append(context_agents['soil'])
    if "crop" in decision:
        selected.append(context_agents['crop'])
    if "economist" in decision or "economics" in decision:
        selected.append(context_agents['economics'])
    
    if not selected:
        selected.append(base_agent)
    
    print(f"🔍 DEBUG: Routing to: {[a.name for a in selected]}")
    return selected


# Orchestrator with memory

async def run_agronomy_team(
    question: str,
    user_email: str,
    user_id: int,
    conversation_id: Optional[int] = None,
    language: Optional[str] = "en",
    images: Optional[List[str]] = None,  
    vision_data: Optional[Dict[str, Any]] = None
) -> str:
    """Run agronomy team with full memory context - optimized for speed"""
    
    # Load persisted memory (fast - just database read)
    try:
        from db import get_user_facts
        user_facts = await get_user_facts(user_id)
        if user_facts:
            facts_lines = "\n".join(f"- {k}: {v}" for k, v in user_facts.items())
            global_memory = f"Known user facts:\n{facts_lines}"
        else:
            global_memory = ""
        print(f"📝 Using user_facts only: {global_memory[:100] if global_memory else 'NONE'}")
    except Exception as e:
        print(f"⚠️ Could not read user facts: {e}")
        global_memory = ""
    
    # Get conversation-specific context (with caching)
    conversation_context = ""
    if conversation_id:
        conversation_context = await get_conversation_context(conversation_id, max_messages=15)

    creator_keywords = [
        "who made you", "who created you", "who built you", "who developed you",
        "who made this", "who created this", "who built this", "who developed this",
        "who is your developer", "who is your creator", "who is behind this",
        "tumhe kisne banaya", "kisne banaya", "developer kaun hai", "aapko kisne banaya"
    ]
    if any(kw in question.lower() for kw in creator_keywords):
        creator_response = """The AI Agronomist was designed and developed by **NCAI -NEDUET, Smart City Lab**.

Feel free to connect or explore more of his work!"""
        return creator_response

    # Build vision context string if available
    vision_context = ""
    if vision_data:
        vision_context = f"""
🔬 VISION ANALYSIS RESULTS:
- Detected Crop: {vision_data.get('crop', 'Unknown')}
- Health Status: {vision_data.get('status', 'Unknown')}
- Condition/Disease: {vision_data.get('label', 'Unknown')}
- Detection Confidence: {vision_data.get('confidence', 0):.1%}
- Analysis Mode: {vision_data.get('mode', 'unknown')}
"""
        if vision_data.get('advice'):
            vision_context += f"- Recommended Action: {vision_data['advice']}\n"
        
        if vision_data.get('mode') == 'auto_uncertain':
            vision_context += "\n⚠️ NOTE: Auto-detection confidence was low. Results are best-guess predictions.\n"
        
        print(f"🔬 Vision context added: {vision_data.get('crop')} - {vision_data.get('label')}")

    # Computed once, reused by every return path below so the diagnosis
    # always surfaces the same way regardless of which branch responds.
    vision_header = _format_vision_header(vision_data)

    # Mirror the user's own language/script rather than forcing English by
    # default: if they type in English, respond in English; if they type in
    # Roman Urdu, respond in Roman Urdu. The explicit "ur" toggle still forces
    # Arabic-script Urdu (via the translation step further down) regardless
    # of what the user types, since that's a deliberate UI choice.
    if language == "roman_ur":
        language_instructions = (
            "⚠️ LANGUAGE REQUIREMENT:\n"
            "- Respond in Roman Urdu using Latin/English letters only.\n"
            "- Do NOT use Urdu script, Hindi, or Devanagari.\n"
        )
    elif language and language.startswith("ur"):
        language_instructions = (
            "⚠️ LANGUAGE REQUIREMENT:\n"
            "- Respond in Urdu using Urdu script (Perso-Arabic/Nastaliq) — the script used to write Urdu in Pakistan.\n"
            "- Do NOT use Hindi or Devanagari script under any circumstances, even though Hindi and Urdu sound similar when spoken — they use completely different scripts and must not be confused.\n"
            "- Do NOT use Roman/Latin letters.\n"
        )
    else:
        language_instructions = (
            "⚠️ LANGUAGE REQUIREMENT:\n"
            "- Mirror the user's language and script exactly.\n"
            "- If the user writes in English, respond in English.\n"
            "- If the user writes in Roman Urdu (Urdu words spelled with English/Latin letters, e.g. 'mujhe iski maloomat chahiye'), respond in Roman Urdu using Latin letters — do not switch to Urdu script or Hindi/Devanagari script.\n"
            "- NEVER respond in Hindi or Devanagari script under any circumstances.\n"
            "- Keep this consistent for the rest of the conversation unless the user changes language.\n"
        )

    # Create personalized base agent
    base_agent = create_base_agent(user_email, global_memory, conversation_context, vision_context, language_instructions)

    # Prepare question with image context
    if images and len(images) > 0:
        image_context = f"\n\n[User has shared {len(images)} image(s) for analysis. Please analyze the visual information and provide relevant agricultural advice.]\n"
        question_for_model = image_context + question
    else:
        question_for_model = question

    # Build message
    user_message = {
        "role": "user",
        "content": question_for_model
    }

    # Responses API item style — required by Runner.run() input
    if images and len(images) > 0:
        user_message["content"] = [
            {"type": "input_text", "text": question_for_model}
        ]
        for img_base64 in images[:4]:  # Limit to 4 images
            user_message["content"].append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{img_base64}"
            })

    messages = [user_message]

    # RAG: check knowledge base first before using Gemini.
    # query_rag() is a synchronous (blocking) DB call — run it in a worker
    # thread so it doesn't block the event loop for other requests.
    _is_vision = vision_data is not None or "[VISION ANALYSIS]" in question
    if _is_vision:
        print(f"🔬 Vision request — skipping RAG, routing to Gemini")
    try:
        from rag import query_rag
        rag_result = await asyncio.to_thread(query_rag, question) if not _is_vision else {"found": False, "context": "", "sources": [], "best_score": 0.0}
        if not _is_vision:
            print(f"📚 RAG {'✅ HIT' if rag_result['found'] else '❌ MISS'} | score: {rag_result['best_score']} | query: {question[:60]}")

        if rag_result["found"]:
            # Zero Gemini calls — return PDF content directly
            response_text = rag_result["context"] 
            print(f"✅ Returning RAG answer directly (0 Gemini calls used)")

            # agr urdu mode on hai tou lang = urdu
            if language and language.startswith("ur"):
                print("translating to urdu using free urdu translation service")
                response_text = translate_to_urdu_free(response_text)
            return response_text

    except Exception as rag_e:
        print(f"⚠️ RAG error (falling back to Gemini): {rag_e}")

    # Fast escalation check (keyword-based, no routing needed)
    urgent_keywords = ["urgent", "emergency", "dying", "rotting", "serious condition", "crisis", "problem", "help immediately"]
    if any(kw in question.lower() for kw in urgent_keywords):
        print(f"🚨 Escalation detected")

        # Use simple string prompt instead of messages list
        escalation_prompt = f"""You are helping with an urgent agricultural issue.
        
    User Context: {global_memory}
    Recent Conversation: {conversation_context}

    User's urgent question: {question}
    {f"[User has provided {len(images)} image(s) showing the issue]" if images else ""}
    Provide immediate, practical advice to address this urgent situation. Be specific and actionable."""

        try:
            result = await _run_with_retry(base_agent, escalation_prompt, max_turns=3)
            response = result.final_output

        except Exception as e:
             print(f"❌ Escalation error: {e}")
             response = "I understand this is urgent. Here's immediate advice:\n\n1. Assess the situation quickly\n2. Take photos if possible\n3. Check for visible pests or diseases\n4. Ensure proper irrigation"

        return f"{vision_header}{response}\n\n📞 Professional Agronomist Contact:\n• Name: Jeetash\n• Phone: +92 3146546335\n• WhatsApp: wa.me/9265326123156\n• Availability: For urgent agricultural assistance"

    # Simplified routing - one LLM call
    router_prompt = f"""Analyze this question and respond with ONE word only: Soil, Crop, Economics, or General.

Question: "{question}"

Your response (one word only):"""

    router_result = await _run_with_retry(base_agent, router_prompt, max_turns=1)
    decision = router_result.final_output.strip().lower()
    print(f"🔍 DEBUG: Routing decision: {decision}")

    # Create only the needed specialized agent (lazy loading)
    if "soil" in decision:
        expert = soil_agent
    elif "crop" in decision:
        expert = crop_agent
    elif "economics" in decision or "economist" in decision:
        expert = economics_agent
    else:
        expert = base_agent

    # Run the selected expert (with retry on transient Gemini overload)
    result = await _run_with_retry(expert, messages, max_turns=5)
    response_text = result.final_output

    # If Urdu requested, translate final output
    if language and language.startswith("ur"):
        translations = await translate_to_urdu_batch([response_text])
        response_text = translations[0]

    # Prepend the guaranteed diagnosis header AFTER translation, so it's never
    # garbled by machine translation and stays consistently formatted.
    if vision_header:
        response_text = vision_header + response_text

    return response_text

# translation
_TRANSLATION_CACHE: Dict[str, str] = {}

async def translate_to_urdu_batch(texts: List[str]) -> List[str]:
    """
    Translate a list of short texts to Urdu (Arabic script) in a single call.
    Uses a dedicated translator agent. On failure, returns originals.
    """
    if not texts:
        return []

    # prepare list of texts that are not cached
    to_translate, indexes = [], []
    results: List[str] = [None] * len(texts)  # type: ignore

    for i, t in enumerate(texts):
        if not t:
            results[i] = t
        elif t in _TRANSLATION_CACHE:
            results[i] = _TRANSLATION_CACHE[t]
        else:
            indexes.append(i)
            to_translate.append(t)

    if not to_translate:
        return results  # type: ignore

    # Build prompt
    prompt_items = "\n".join(f"{idx}: {text.replace(chr(10),' ')}" for idx, text in enumerate(to_translate))
    prompt = (
        "Translate the following items to Urdu using Arabic script. "
        "Return a JSON array of translations in the same order, and return only valid JSON.\n\n"
        f"Items:\n{prompt_items}\n\nOutput JSON:"
    )

    try:
        translator_agent = Agent(
            name="TranslatorAgent",
            instructions=(
                "You are a concise translator specializing in Urdu as written in Pakistan. "
                "Translate the provided items into Urdu using Perso-Arabic (Nastaliq) script only — the script used for Urdu in Pakistan. "
                "Do NOT use Hindi or Devanagari script under any circumstances — this is a common mistake since Hindi and Urdu are spoken almost identically but must never share a script. "
                "Do NOT use Roman/Latin letters either. "
                "Return only a JSON array of translated strings in the same order as the input. Do not add commentary."
            ),
            tools=[],
            model=ctx.model,
            model_settings=ModelSettings(temperature=0.0),
            handoffs=[],
        )

        res = await _run_with_retry(translator_agent, [{"role": "user", "content": prompt}], max_turns=1)
        raw = (getattr(res, "final_output", None) or "").strip()

        if raw.startswith("```json") and raw.endswith("```"):
            raw = raw[7:-3].strip()
        elif raw.startswith("```") and raw.endswith("```"):
            raw = raw[3:-3].strip()

        # try to parse JSON
        try:
            translations = json.loads(raw)
            if not isinstance(translations, list):
                raise ValueError("Expected JSON array")
        except Exception:
            translations = [line.strip() for line in raw.splitlines() if line.strip()]

        # fill results and cache
        for idx, text_index in enumerate(indexes):
            translated = translations[idx] if idx < len(translations) else to_translate[idx]
            _TRANSLATION_CACHE[to_translate[idx]] = translated
            results[text_index] = translated

        return results  # type: ignore

    except Exception as e:
        print("translate_to_urdu_batch error:", e)
        return texts


# API-facing processing

async def process_with_agronomy_team(message: str, email: str, conversation_id: Optional[int] = None, language: Optional[str] = "en", images: Optional[List[str]] = None, vision_data: Optional[Dict[str, Any]] = None) -> str:
    """Process messages with full memory context and image support
    language: 'en' or 'ur' (or other codes) — 'ur' will instruct the agent to reply in Urdu (Arabic script).
    """
    try:
        #Guardrail
        if check_for_abuse(message):
            return handle_abuse_response()

        user = await async_get_or_create_user(email, email.split('@')[0])

        if (
            vision_data
            and vision_data.get("mode") == "auto_uncertain"
            and images
            and len(images) > 0
        ):
            msg_lower = message.lower()
            confirmed_crop = None
            for crop in ("wheat", "corn", "rice", "cotton"):
                if crop in msg_lower:
                    confirmed_crop = crop
                    break
            if confirmed_crop:
                print(f"[agronomy] User confirmed crop: {confirmed_crop}")
                from vision import predict_crop_from_base64, _get_advice_for_label

                result = predict_crop_from_base64(images[0], confirmed_crop)
                advice = _get_advice_for_label(confirmed_crop, result["label"])
                result["advice"] = advice
                result["mode"] = "manual"
                result["crop_confidence"] = vision_data.get("crop_confidence", 0.0)
                vision_data = result  # ✅ Replace with confirmed analysis
        response = await run_agronomy_team(
            question=message,
            user_email=email,
            user_id=user.id,
            conversation_id=conversation_id,
            language=language,
            images=images,
            vision_data=vision_data
        )
        return response
    except Exception as e:
        error_str = str(e)
        print(f"❌ Error in process_with_agronomy_team: {error_str}")

        is_quota = "RESOURCE_EXHAUSTED" in error_str or "429" in error_str or "quota" in error_str.lower()
        is_overloaded = "UNAVAILABLE" in error_str or "503" in error_str or "high demand" in error_str.lower()

        if is_quota or is_overloaded:
            # Retries in _run_with_retry already handled short-lived overload —
            # reaching here means either quota is exhausted, or overload persisted
            # past the retry window. Still surface the vision diagnosis if we have
            # one, since that part completed successfully and cost nothing extra.
            if vision_data and vision_data.get("mode") in ("auto", "manual"):
                crop = vision_data.get("crop", "your crop")
                label = vision_data.get("label", "unknown")
                advice = vision_data.get("advice", "")
                confidence = vision_data.get("confidence", 0)

                if is_quota:
                    notice = "⚠️ Our AI assistant has hit its daily usage limit, but here's what the vision analysis found:"
                    followup = "Please try again later for a more detailed conversational response."
                else:
                    notice = "⚠️ Our AI assistant is temporarily overloaded, but here's what the vision analysis found:"
                    followup = "This usually resolves within a minute — please try sending your message again shortly for a full response."

                return (
                    f"{notice}\n\n"
                    f"🌾 **{crop} — {label}**\n"
                    f"_Confidence: {confidence:.0%}_\n\n"
                    f"{advice}\n\n"
                    f"{followup}"
                )

            if is_quota:
                return "⚠️ Our AI assistant has reached its daily usage limit for now. Please try again in a little while — this usually resets within 24 hours."
            return "⚠️ Our AI assistant is temporarily overloaded (high demand on Google's end). Please try again in a minute."

        return "I apologize, but I'm experiencing technical difficulties. Please try again."

async def main() -> None:
    print("🌱 Agronomy Deep Research Agent ready! Type 'exit' to quit.\n")
    await init_db()
    user_email = input("👤 Enter your email address: ").strip() or "guest@example.com"

    user = await async_get_or_create_user(user_email, user_email.split('@')[0])
    print(f"✅ Welcome, {user.name}! Your chats will be saved under user_id={user.id}\n")

    # Use database for conversation tracking
    from db import async_create_conversation
    conversation_id = await async_create_conversation(user.id, "Terminal Session")

    while True:
        try:
            question = input("❓ Ask your agronomy question: ").strip()
            if question.lower() in ("exit", "quit"):
                print("👋 Goodbye!")
                break

            if question.startswith("/history"):
                try:
                    limit = int(question.split(" ")[1])
                except (IndexError, ValueError):
                    limit = 10
                history = await async_get_history(user.id, limit)
                print("\n📜 Your recent chat history:")
                for h in history:
                    role = "👤 You" if h.role == 'user' else "🤖 Agronomist"
                    print(f"{role}: {h.content}")
                print("")
                continue

            if question.startswith("/new"):
                conversation_id = await async_create_conversation(user.id, "Terminal Session")
                print("🆕 New conversation started!")
                continue

            print("\n💬 Thinking...\n")
            team_result = await run_agronomy_team(question, user_email, user.id, conversation_id)

            # Save to database
            await async_save_message(user.id, "user", question, conversation_id)
            await async_save_message(user.id, "assistant", team_result, conversation_id)

            print(team_result)
            print("\n" + "="*50 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Session ended.")
            break

if __name__ == "__main__":
    asyncio.run(main())