import asyncio
import base64
import os
import re
import tempfile
from io import BytesIO

import aiohttp
from PIL import Image as PILImage

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Reply
from astrbot.api.star import Context, Star, register

# ── 支持的比例列表 ──

SUPPORTED_RATIOS = [
    "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3",
    "5:4", "4:5", "21:9", "9:21", "1:2", "2:1",
    "1:3", "3:1", "1:4", "4:1", "1:8", "8:1",
]

RATIO_RE = re.compile(r"^(\d+)[:：](\d+)$")
PIXEL_RE = re.compile(r"^(\d+)[x×](\d+)$", re.IGNORECASE)

# Grsai 文档中列出的常用像素尺寸。新版部分模型（例如 gpt-image-2-vip）
# 可能不再接受 "1:1" 这类比例字符串，而要求传入 "1024x1024"。
# 这里按输出分辨率档位保留一份映射，旧版仍可继续使用比例字符串。
PIXEL_SIZE_MAP = {
    "1K": {
        "1:1": "1024x1024", "16:9": "1280x720", "9:16": "720x1280",
        "4:3": "1152x864", "3:4": "864x1152", "3:2": "1248x832",
        "2:3": "832x1248", "5:4": "1120x896", "4:5": "896x1120",
        "21:9": "1456x624", "9:21": "624x1456",
    },
    "2K": {
        "1:1": "2048x2048", "16:9": "2560x1440", "9:16": "1440x2560",
        "4:3": "2304x1728", "3:4": "1728x2304", "3:2": "2496x1664",
        "2:3": "1664x2496", "5:4": "2240x1792", "4:5": "1792x2240",
        "21:9": "3024x1296", "9:21": "1296x3024",
    },
    "4K": {
        "1:1": "2880x2880", "16:9": "3840x2160", "9:16": "2160x3840",
        "4:3": "3264x2448", "3:4": "2448x3264", "3:2": "3504x2336",
        "2:3": "2336x3504", "5:4": "3200x2560", "4:5": "2560x3200",
        "21:9": "3696x1584", "9:21": "1584x3696",
    },
}


def _ratio_to_float(ratio_str: str) -> float | None:
    try:
        w_str, h_str = ratio_str.split(":")
        return int(w_str) / int(h_str)
    except (ValueError, ZeroDivisionError):
        return None


@register(
    "astrbot_plugin_grsai",
    "Bigesila-B",
    "Grsai 图像生成插件",
    "1.0.0",
    "https://github.com/Bigesila-B/astrbot_plugin_grsai",
)
class GrsaiPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._models = self._parse_models()
        self._presets = self._parse_presets()

    # ── 模型解析 ──

    def _parse_models(self) -> list:
        raw = self.config.get("models", [])
        if isinstance(raw, list):
            return [m for m in raw if isinstance(m, dict) and m.get("model_id")]
        return []

    def _default_model(self) -> dict:
        """返回第一个模型配置作为默认，无配置时返回空 dict"""
        return self._models[0] if self._models else {}

    # ── 预设解析 ──

    def _parse_presets(self) -> list:
        raw = self.config.get("presets", [])
        if isinstance(raw, list):
            return [p for p in raw if isinstance(p, dict) and p.get("trigger")]
        return []

    def _find_preset(self, message_str: str) -> dict | None:
        for preset in self._presets:
            trigger = preset.get("trigger", "")
            if trigger and trigger in message_str:
                return preset
        return None

    def _save_presets(self):
        self.config["presets"] = self._presets
        try:
            self.config.save_config()
        except AttributeError:
            pass

    # ── 消息监听 ──

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """监听所有消息，按模型配置的触发词路由"""
        message_str = event.message_str.strip()

        # 检查各模型的文生图 / 图生图触发词
        for model in self._models:
            t2i = model.get("trigger_t2i", "GPT生图").strip()
            i2i = model.get("trigger_i2i", "GPT改图").strip()

            if t2i and t2i in message_str:
                async for result in self._handle_text_to_image(event, message_str, model, t2i):
                    yield result
                event.stop_event()
                return

            if i2i and i2i in message_str:
                async for result in self._handle_image_edit(event, message_str, model, i2i):
                    yield result
                event.stop_event()
                return

        # ── 预设管理命令 ──

        if message_str.startswith("列出预设"):
            async for result in self._handle_list_presets(event):
                yield result
            event.stop_event()
            return

        if message_str.startswith("删除预设"):
            async for result in self._handle_delete_preset(event, message_str):
                yield result
            event.stop_event()
            return

        if message_str.startswith("添加预设"):
            async for result in self._handle_add_preset(event, message_str):
                yield result
            event.stop_event()
            return

        # 检查自定义预设触发词
        preset = self._find_preset(message_str)
        if preset:
            async for result in self._handle_preset(event, message_str, preset):
                yield result
            event.stop_event()
            return

    # ── 预设管理命令 ──

    async def _handle_add_preset(self, event: AstrMessageEvent, message_str: str):
        try:
            payload = message_str.split("添加预设", 1)[1]
        except IndexError:
            yield event.plain_result(
                "格式：添加预设 触发词|LLM地址|APIKey|模型ID|预设提示词\n"
                "示例：添加预设 动漫化|https://api.openai.com/v1|sk-xxx|gpt-4o|分析图片生成动漫风格提示词"
            )
            return
        payload = payload.strip()
        if not payload:
            yield event.plain_result(
                "格式：添加预设 触发词|LLM地址|APIKey|模型ID|预设提示词\n"
                "示例：添加预设 动漫化|https://api.openai.com/v1|sk-xxx|gpt-4o|分析图片生成动漫风格提示词"
            )
            return
        parts = payload.split("|", 4)
        if len(parts) < 5:
            yield event.plain_result("❌ 参数不足，需要 5 个参数（用 | 分隔）")
            return
        trigger, llm_url, llm_apikey, llm_model, preset_prompt = [p.strip() for p in parts]
        if not trigger:
            yield event.plain_result("❌ 触发词不能为空")
            return
        for p in self._presets:
            if p.get("trigger") == trigger:
                yield event.plain_result(f"❌ 触发词「{trigger}」已存在，请先删除再添加")
                return
        self._presets.append({
            "trigger": trigger,
            "llm_url": llm_url,
            "llm_apikey": llm_apikey,
            "llm_model": llm_model,
            "preset_prompt": preset_prompt,
        })
        self._save_presets()
        yield event.plain_result(f"✅ 预设「{trigger}」已添加")

    async def _handle_delete_preset(self, event: AstrMessageEvent, message_str: str):
        try:
            trigger = message_str.split("删除预设", 1)[1].strip()
        except IndexError:
            yield event.plain_result("格式：删除预设 触发词\n示例：删除预设 动漫化")
            return
        if not trigger:
            yield event.plain_result("❌ 请输入要删除的触发词")
            return
        for i, p in enumerate(self._presets):
            if p.get("trigger") == trigger:
                del self._presets[i]
                self._save_presets()
                yield event.plain_result(f"✅ 预设「{trigger}」已删除")
                return
        yield event.plain_result(f"❌ 未找到触发词「{trigger}」的预设")

    async def _handle_list_presets(self, event: AstrMessageEvent):
        if not self._presets:
            yield event.plain_result("📋 当前没有自定义预设")
            return
        lines = ["📋 当前预设："]
        for p in self._presets:
            trigger = p.get("trigger", "?")
            model = p.get("llm_model", "?")
            prompt_preview = p.get("preset_prompt", "")[:40]
            lines.append(f"  • {trigger} → {model} | {prompt_preview}...")
        yield event.plain_result("\n".join(lines))

    # ── 指令处理 ──

    async def _handle_text_to_image(self, event: AstrMessageEvent, message_str: str,
                                    model: dict, trigger: str):
        try:
            text = message_str.split(trigger, 1)[1]
        except IndexError:
            yield event.plain_result(
                f"❌ 请提供生成提示词\n格式：{trigger} [比例] 提示词\n示例：{trigger} 2:3 一只猫")
            return
        prompt, custom_ratio = self._parse_prompt_and_ratio(text)
        if not prompt:
            yield event.plain_result(
                f"❌ 请提供生成提示词\n格式：{trigger} [比例] 提示词\n示例：{trigger} 2:3 一只猫")
            return
        aspect_ratio = self._resolve_aspect_ratio(custom_ratio, model)
        yield event.plain_result(f"⏳ 正在生成图片（{aspect_ratio}），请稍候...")
        result, retries = await self._call_api_with_violation_retry(
            prompt, images=[], aspect_ratio=aspect_ratio, model=model)
        if retries > 0:
            yield event.plain_result(f"🔄 内容安全审核触发，已自动重试 {retries} 次")
        if result.get("status") == "succeeded" and result.get("results"):
            async for r in self._send_image_result(event, result["results"][0]["url"]):
                yield r
        else:
            yield event.plain_result(f"❌ 生成失败：{result.get('error') or '未知错误'}")

    async def _handle_image_edit(self, event: AstrMessageEvent, message_str: str,
                                 model: dict, trigger: str):
        try:
            text = message_str.split(trigger, 1)[1]
        except IndexError:
            yield event.plain_result(
                f"❌ 请提供改图提示词和图片\n格式：{trigger} [比例] 提示词 + 图片")
            return
        prompt, custom_ratio = self._parse_prompt_and_ratio(text)
        images = await self._extract_images(event)
        if not prompt:
            yield event.plain_result(
                f"❌ 请提供改图提示词\n格式：{trigger} [比例] 提示词 + 图片")
            return
        if not images:
            yield event.plain_result(
                f"❌ 请附带至少一张图片\n格式：{trigger} [比例] 提示词 + 图片")
            return
        if custom_ratio:
            aspect_ratio = self._resolve_aspect_ratio(custom_ratio, model)
        else:
            img_ratio = await self._get_image_ratio_from_base64(images[0], model)
            if img_ratio:
                aspect_ratio = self._resolve_aspect_ratio(img_ratio, model)
                yield event.plain_result(f"📐 检测到原图比例接近 {img_ratio}，按 {aspect_ratio} 生成")
            else:
                aspect_ratio = self._resolve_aspect_ratio(None, model)
        yield event.plain_result(f"⏳ 正在处理图片（共 {len(images)} 张，{aspect_ratio}），请稍候...")
        result, retries = await self._call_api_with_violation_retry(
            prompt, images=images, aspect_ratio=aspect_ratio, model=model)
        if retries > 0:
            yield event.plain_result(f"🔄 内容安全审核触发，已自动重试 {retries} 次")
        if result.get("status") == "succeeded" and result.get("results"):
            async for r in self._send_image_result(event, result["results"][0]["url"]):
                yield r
        else:
            yield event.plain_result(f"❌ 改图失败：{result.get('error') or '未知错误'}")

    # ── 比例解析 ──

    def _parse_prompt_and_ratio(self, text: str) -> tuple:
        text = text.strip()
        if not text:
            return "", None
        parts = text.split(None, 1)
        first_word = parts[0]
        match = RATIO_RE.match(first_word)
        if match:
            ratio = f"{match.group(1)}:{match.group(2)}"
            prompt = parts[1].strip() if len(parts) > 1 else ""
            return prompt, ratio
        return text, None

    def _pixel_mode_enabled(self, model: dict | None) -> bool:
        """统一解析配置中的比例转像素开关，兼容布尔值和字符串值。"""
        value = (model or {}).get("pixel_size_mode", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "像素", "像素模式"}
        return bool(value)

    def _resolve_aspect_ratio(self, custom_ratio: str | None, model: dict = None) -> str:
        """解析用户输入，并按模型配置决定传比例还是像素值。

        默认保持旧行为：直接把 "1:1"、"16:9" 等比例传给 API。
        对新版严格要求像素值的模型，可在模型配置中开启 ``pixel_size_mode``，
        插件会自动把比例转换为对应档位的具体尺寸，例如 1:1 -> 1024x1024。
        用户直接输入 1024x1024 等像素值时始终原样保留。
        """
        model = model or {}
        value = (custom_ratio or self.config.get("aspect_ratio", "1:1")).strip()
        if not value:
            value = "1:1"

        # 像素格式直接透传，兼容用户手动指定具体尺寸的旧/新 API。
        if PIXEL_RE.match(value):
            return value.replace("×", "x")

        # 非比例格式也不擅自改写，避免破坏第三方 API 的扩展值（如 auto）。
        match = RATIO_RE.match(value)
        if not match:
            return value
        ratio = f"{match.group(1)}:{match.group(2)}"

        # pixel_size_mode 可填 true/false，也兼容字符串配置。
        if self._pixel_mode_enabled(model):
            resolution = str(model.get("image_size", "1K")).upper()
            return PIXEL_SIZE_MAP.get(resolution, PIXEL_SIZE_MAP["1K"]).get(ratio, ratio)

        return ratio

    async def _get_image_ratio_from_base64(self, base64_uri: str, model: dict = None) -> str | None:
        try:
            if "," in base64_uri:
                base64_uri = base64_uri.split(",", 1)[1]
            data = base64.b64decode(base64_uri)
            img = PILImage.open(BytesIO(data))
            w, h = img.size
            if h == 0:
                return None
            return self._find_closest_ratio(w, h, model)
        except Exception as e:
            logger.error(f"解析图片尺寸失败: {e}")
            return None

    def _find_closest_ratio(self, width: int, height: int, model: dict = None) -> str:
        """根据图片宽高，在支持的比例中找到最接近的"""
        target = width / height if height else 1.0
        best_ratio = "1:1"
        best_diff = float("inf")
        for ratio_str in SUPPORTED_RATIOS:
            val = _ratio_to_float(ratio_str)
            if val is None:
                continue
            diff = abs(val - target)
            if diff < best_diff:
                best_diff = diff
                best_ratio = ratio_str
        return best_ratio

    # ── 图片提取 ──

    async def _extract_images(self, event: AstrMessageEvent) -> list:
        return await self._extract_images_from_chain(event.get_messages())

    async def _extract_images_from_chain(self, chain: list) -> list:
        """递归从消息链中提取图片（含引用消息中的图片）"""
        images = []
        for comp in chain:
            if isinstance(comp, Image):
                b64 = await self._image_to_base64_data_uri(comp)
                if b64:
                    images.append(b64)
            elif isinstance(comp, Reply):
                for attr in ("chain", "message", "origin", "content"):
                    nested_chain = getattr(comp, attr, None)
                    if nested_chain and isinstance(nested_chain, list):
                        nested_images = await self._extract_images_from_chain(nested_chain)
                        if nested_images:
                            images.extend(nested_images)
                            break
        logger.info(f"图片提取: 从消息链中提取到 {len(images)} 张图片")
        return images

    async def _image_to_base64_data_uri(self, img: Image) -> str:
        """将 AstrBot Image 组件转为 base64 data URI"""
        # 优先使用官方方法
        try:
            b64 = await img.convert_to_base64()
            if b64:
                return f"data:image/png;base64,{b64}"
        except Exception:
            pass
        # 回退：手动处理 url / file
        if getattr(img, "file", None):
            b64 = await self._file_as_base64(img.file)
            if b64:
                return b64
        if getattr(img, "url", None):
            b64 = await self._download_as_base64(img.url)
            if b64:
                return b64
        logger.warning(f"图片提取: 无法处理 Image 组件, file={getattr(img, 'file', None)}, url={getattr(img, 'url', None)}")
        return ""

    async def _download_as_base64(self, url: str) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        logger.error(f"下载图片失败, status={resp.status}")
                        return ""
                    content_type = resp.headers.get("Content-Type", "image/png")
                    data = await resp.read()
                    b64 = base64.b64encode(data).decode("utf-8")
                    return f"data:{content_type};base64,{b64}"
        except Exception as e:
            logger.error(f"下载图片异常: {e}")
            return ""

    async def _file_as_base64(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            logger.error(f"图片文件不存在: {file_path}")
            return ""
        ext = os.path.splitext(file_path)[1].lower()
        ext_map = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".gif": "gif", ".webp": "webp"}
        mime_type = f"image/{ext_map.get(ext, 'png')}"
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode("utf-8")
            return f"data:{mime_type};base64,{b64}"
        except Exception as e:
            logger.error(f"读取图片文件异常: {e}")
            return ""

    # ── 图片压缩 ──

    def _compress_for_llm(self, base64_uri: str) -> str:
        try:
            if "," in base64_uri:
                header, b64 = base64_uri.split(",", 1)
            else:
                header = "data:image/jpeg"
                b64 = base64_uri
            data = base64.b64decode(b64)
            original_size = len(data)
            img = PILImage.open(BytesIO(data))
            w, h = img.size
            needs_resize = w > 1024 or h > 1024
            needs_compress = original_size > 1024 * 1024
            if not needs_resize and not needs_compress:
                return base64_uri
            if needs_resize:
                ratio = 1024.0 / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), PILImage.LANCZOS)
            if needs_compress or needs_resize:
                quality = 80
                while quality >= 30:
                    buf = BytesIO()
                    img.convert("RGB").save(buf, format="JPEG", quality=quality)
                    if buf.tell() <= 1024 * 1024:
                        break
                    quality -= 10
                buf.seek(0)
                compressed_b64 = base64.b64encode(buf.read()).decode("utf-8")
                logger.info(f"图片压缩: {w}x{h} → {img.size[0]}x{img.size[1]}, "
                            f"{original_size / 1024:.0f}KB → {buf.tell() / 1024:.0f}KB, q={quality}")
                return f"data:image/jpeg;base64,{compressed_b64}"
            return base64_uri
        except Exception as e:
            logger.error(f"图片压缩失败: {e}")
            return base64_uri

    # ── 预设工作流 ──

    async def _handle_preset(self, event: AstrMessageEvent, message_str: str, preset: dict):
        trigger = preset["trigger"]
        llm_url = preset.get("llm_url", "").strip()
        llm_apikey = preset.get("llm_apikey", "").strip()
        llm_model = preset.get("llm_model", "gpt-4o").strip()
        preset_prompt = preset.get("preset_prompt", "").strip()
        model_trigger = preset.get("model_trigger", "").strip()

        if not llm_url:
            yield event.plain_result(f"❌ 预设「{trigger}」未配置 LLM 地址")
            return
        if not llm_apikey:
            yield event.plain_result(f"❌ 预设「{trigger}」未配置 LLM API Key")
            return
        if not preset_prompt:
            yield event.plain_result(f"❌ 预设「{trigger}」未配置提示词预设")
            return

        # 选择生图模型：按 model_trigger 匹配，否则默认第一个
        gen_model = self._default_model()
        if model_trigger:
            for m in self._models:
                if (m.get("trigger_t2i", "") == model_trigger or
                        m.get("trigger_i2i", "") == model_trigger):
                    gen_model = m
                    break
        if not gen_model:
            yield event.plain_result("❌ 未配置生图模型，请先在插件设置中添加模型")
            return

        try:
            extra_text = message_str.split(trigger, 1)[1].strip()
        except IndexError:
            extra_text = ""

        images = await self._extract_images(event)
        if not images:
            yield event.plain_result(f"❌ 请附带至少一张图片\n格式：{trigger} [附加说明] + 图片")
            return

        llm_prompt = f"{preset_prompt}\n\n用户附加要求：{extra_text}" if extra_text else preset_prompt
        llm_images = [self._compress_for_llm(img) for img in images]

        yield event.plain_result(f"⏳ 正在生成「{trigger}」，请稍候...")

        llm_result = await self._call_llm_vision(llm_url, llm_apikey, llm_model, llm_prompt, llm_images)

        if llm_result.startswith("LLM 错误") or llm_result.startswith("LLM 调用失败"):
            yield event.plain_result(f"❌ {llm_result}")
            return
        if not llm_result or not llm_result.strip():
            yield event.plain_result("❌ 语言模型未返回有效提示词，请检查 LLM 配置或图片是否可识别")
            return

        # 按原图比例生成
        img_ratio = await self._get_image_ratio_from_base64(images[0], gen_model)
        aspect_ratio = self._resolve_aspect_ratio(img_ratio, gen_model) if img_ratio else self._resolve_aspect_ratio(None, gen_model)

        # 调用生图模型改图
        result, retries = await self._call_api_with_violation_retry(
            llm_result, images=images, aspect_ratio=aspect_ratio, model=gen_model)

        if retries > 0:
            yield event.plain_result(f"🔄 内容安全审核触发，已自动重试 {retries} 次")
        if result.get("status") == "succeeded" and result.get("results"):
            async for r in self._send_image_result(event, result["results"][0]["url"]):
                yield r
        else:
            yield event.plain_result(f"❌ 生成失败：{result.get('error') or '未知错误'}")

    async def _call_llm_vision(self, llm_url: str, apikey: str, model: str,
                               prompt: str, images: list) -> str:
        url = llm_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {apikey}", "Content-Type": "application/json"}
        content = [{"type": "text", "text": prompt}]
        for img_b64 in images:
            content.append({"type": "image_url", "image_url": {"url": img_b64}})
        payload = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 2048}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    result = await resp.json()
                    logger.info(f"LLM API 响应: has_choices={'choices' in result}, "
                                f"model={model}, images={len(images)}")
                    if "choices" in result and len(result["choices"]) > 0:
                        msg = result["choices"][0].get("message", {})
                        content = (msg.get("content") or
                                   msg.get("reasoning_content") or "")
                        content = content.strip()
                        if content:
                            return content
                        logger.warning(f"LLM 返回空内容: choices={result['choices'][0]}")
                        return "LLM 错误：模型未返回有效文本（可能拒绝处理或内容被过滤）"
                    err = result.get("error", {})
                    err_msg = err.get("message", "") if isinstance(err, dict) else str(err)
                    return f"LLM 错误：{err_msg or '未知错误'}"
        except aiohttp.ClientError as e:
            logger.error(f"LLM 请求异常: {e}")
            return f"LLM 调用失败：{e}"
        except Exception as e:
            logger.error(f"LLM 调用异常: {e}")
            return f"LLM 调用失败：{e}"

    # ── 图片发送 ──

    async def _send_image_result(self, event: AstrMessageEvent, image_url: str):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        logger.error(f"下载生成图片失败, status={resp.status}")
                        yield event.image_result(image_url)
                        return
                    data = await resp.read()
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                        f.write(data)
                        temp_path = f.name
                    yield event.chain_result([Image.fromFileSystem(temp_path)])
        except Exception as e:
            logger.error(f"发送图片异常: {e}")
            yield event.image_result(image_url)

    # ── API 调用 ──

    def _is_pixel_size_error(self, result: dict) -> bool:
        """判断 API 是否明确要求使用具体像素尺寸。"""
        if not isinstance(result, dict):
            return False
        error_parts = [
            result.get("error"), result.get("message"), result.get("detail"),
        ]
        error = result.get("error")
        if isinstance(error, dict):
            error_parts.extend([error.get("message"), error.get("code")])
        text = " ".join(str(item) for item in error_parts if item).lower()
        return (
            "does not support" in text and "parameter" in text and "pixel" in text
        ) or "please use pixel value" in text

    async def _call_api_with_violation_retry(
        self, prompt: str, images: list, aspect_ratio: str = None, model: dict = None,
    ) -> tuple:
        model = model or self._default_model()
        max_retries = model.get("retry_on_violation", 2)
        if not isinstance(max_retries, int) or max_retries < 0:
            max_retries = 2
        result = await self._call_api(prompt, images, aspect_ratio, model)

        # 新版 gpt-image-2 / gpt-image-2-vip 节点可能拒绝比例字符串。
        # 先按旧规则调用；仅在 API 明确返回“请使用像素值”时自动补发一次，
        # 因此无需改动旧模型配置，也不会对普通失败进行盲目重试。
        if (
            self._is_pixel_size_error(result)
            and aspect_ratio
            and RATIO_RE.match(aspect_ratio)
            and not self._pixel_mode_enabled(model)
        ):
            pixel_ratio = self._resolve_aspect_ratio(aspect_ratio, {**model, "pixel_size_mode": True})
            if pixel_ratio != aspect_ratio:
                logger.info(f"API 要求像素尺寸，自动重试: {aspect_ratio} -> {pixel_ratio}")
                result = await self._call_api(prompt, images, pixel_ratio, model)

        retries = 0
        while result.get("status") == "violation" and retries < max_retries:
            retries += 1
            logger.info(f"内容安全违规，正在第 {retries} 次重试...")
            await asyncio.sleep(1.5)
            result = await self._call_api(prompt, images, aspect_ratio, model)
        if result.get("status") == "violation":
            result["error"] = "内容安全审核未通过，请修改提示词或图片后重试"
        return result, retries

    async def _call_api(self, prompt: str, images: list, aspect_ratio: str = None,
                        model: dict = None) -> dict:
        model = model or self._default_model()
        url = model.get("url", "").strip()
        apikey = model.get("apikey", "").strip()
        model_id = model.get("model_id", "nano-banana-2")
        reply_type = model.get("reply_type", "json")
        image_size = model.get("image_size", "1K")

        if aspect_ratio is None:
            aspect_ratio = self.config.get("aspect_ratio", "1:1")

        if not url:
            return {"status": "failed", "error": "未配置 API URL"}
        if not apikey:
            return {"status": "failed", "error": "未配置 API Key"}
        if not prompt or not prompt.strip():
            return {"status": "failed", "error": "提示词为空，无法生成图片"}

        headers = {"Authorization": f"Bearer {apikey}", "Content-Type": "application/json"}
        payload = {
            "model": model_id, "prompt": prompt, "images": images,
            "aspectRatio": aspect_ratio, "imageSize": image_size,
            "replyType": reply_type,
        }
        try:
            api_url = f"{url.rstrip('/')}/v1/api/generate"
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    api_url, json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    result = await resp.json()
                    logger.info(f"Grsai API: status={result.get('status')}, "
                                f"model={model_id}, imgs={len(images)}, prompt_len={len(prompt)}")
                    if reply_type == "async" and result.get("status") == "running":
                        task_id = result.get("id")
                        if not task_id:
                            return result
                        return await self._poll_async(url, apikey, task_id, model)
                    return result
        except aiohttp.ClientError as e:
            logger.error(f"API 请求异常: {e}")
            return {"status": "failed", "error": f"网络请求失败：{e}"}
        except Exception as e:
            logger.error(f"API 调用异常: {e}")
            return {"status": "failed", "error": f"调用失败：{e}"}

    async def _poll_async(self, base_url: str, apikey: str, task_id: str,
                          model: dict = None) -> dict:
        model = model or self._default_model()
        timeout_sec = model.get("async_timeout", 120)
        if not isinstance(timeout_sec, (int, float)) or timeout_sec <= 0:
            timeout_sec = 120
        interval = 2.0
        max_retries = max(1, int(timeout_sec / interval))
        headers = {"Authorization": f"Bearer {apikey}"}
        poll_url = f"{base_url.rstrip('/')}/v1/api/result"
        for attempt in range(max_retries):
            await asyncio.sleep(interval)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        poll_url, params={"id": task_id}, headers=headers,
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        result = await resp.json()
                        if result.get("status") in ("succeeded", "failed", "violation"):
                            return result
            except Exception as e:
                logger.error(f"轮询异常 (尝试 {attempt + 1}/{max_retries}): {e}")
        return {"status": "failed", "error": f"任务超时（{timeout_sec} 秒）"}

    async def terminate(self):
        pass
