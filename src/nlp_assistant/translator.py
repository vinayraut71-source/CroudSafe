"""
Multilingual Alert Translator supporting English, Hindi (हिन्दी), and Marathi (मराठी).
Guarantees strict preservation of numbers, percentages, timestamps, and alert classifications.
"""

import re
import os
import json
from typing import Dict, Any, List, Tuple, Optional


class AlertTranslator:
    """
    Translates crowd safety alerts between English, Hindi, and Marathi.
    Guarantees that numbers, timestamps, and classifications are preserved without modification.
    """

    # Core phrase translations mapping English patterns to Hindi and Marathi
    DICTIONARY = {
        # General Status & Thresholds
        "all metrics within normal parameters.": {
            "hi": "सभी पैरामीटर सामान्य सीमा के भीतर हैं।",
            "mr": "सर्व मापदंड सामान्य मर्यादेत आहेत."
        },
        "all metrics within normal parameters": {
            "hi": "सभी पैरामीटर सामान्य सीमा के भीतर",
            "mr": "सर्व मापदंड सामान्य मर्यादेत"
        },
        "one or more metrics approaching threshold.": {
            "hi": "एक या अधिक पैरामीटर सीमा के करीब पहुंच रहे हैं।",
            "mr": "एक किंवा अधिक मापदंड मर्यादेच्या जवळ पोहोचत आहेत."
        },
        "multiple metrics in danger zone — monitor closely.": {
            "hi": "खतरे के क्षेत्र में कई पैरामीटर — बारीकी से निगरानी करें।",
            "mr": "धोकादायक क्षेत्रात अनेक मापदंड — बारकाईने लक्ष ठेवा."
        },
        "stampede risk detected — immediate action required!": {
            "hi": "भगदड़ का जोखिम पता चला — तत्काल कार्रवाई आवश्यक!",
            "mr": "चेंगराचेंगरीचा धोका आढळला — तातडीने कारवाई आवश्यक!"
        },
        "multiple metrics elevated": {
            "hi": "कई पैरामीटर बढ़े हुए हैं",
            "mr": "अनेक मापदंड वाढलेले आहेत"
        },
        "crowd density is increasing. please maintain safe distances.": {
            "hi": "भीड़ का घनत्व बढ़ रहा है। कृपया सुरक्षित दूरी बनाए रखें।",
            "mr": "गर्दीची घनता वाढत आहे. कृपया सुरक्षित अंतर ठेवा."
        },
        "warning: crowd levels are rising. security personnel, please monitor the area.": {
            "hi": "चेतावनी: भीड़ का स्तर बढ़ रहा है। सुरक्षा कर्मी, कृपया क्षेत्र की निगरानी करें।",
            "mr": "चेतावणी: गर्दीची पातळी वाढत आहे. सुरक्षा कर्मचारी, कृपया परिसरावर लक्ष ठेवा."
        },
        "danger alert. crowd congestion detected. please move away from crowded areas immediately.": {
            "hi": "खतरा अलर्ट। भीड़भाड़ का पता चला। कृपया भीड़भाड़ वाले इलाके से तुरंत दूर जाएं।",
            "mr": "धोका इशारा. गर्दीची कोंडी आढळली. कृपया गर्दीच्या ठिकाणाहून त्वरित दूर जा."
        },
        "crowd pressure is critical. security team, please respond to the flagged zone.": {
            "hi": "भीड़ का दबाव गंभीर है। सुरक्षा दल, कृपया चिह्नित ज़ोन में तुरंत पहुंचे।",
            "mr": "गर्दीचा दबाव गंभीर आहे. सुरक्षा दल, कृपया चिन्हांकित झोनमध्ये त्वरित पोहोचा."
        },
        "emergency. stampede risk detected. please remain calm and move to the nearest exit quickly.": {
            "hi": "आपातकालीन स्थिति। भगदड़ के जोखिम का संकेत। शांत रहें और निकटतम निकास द्वार की ओर बढ़ें।",
            "mr": "आणीबाणी. चेंगराचेंगरीचा धोका आढळला. शांत राहा आणि जवळच्या बाहेर पडण्याच्या मार्गाकडे त्वरा करा."
        },
        "critical crowd compression. all security personnel respond immediately. emergency protocol activated.": {
            "hi": "अत्यधिक भीड़ संपीड़न। सभी सुरक्षा कर्मी तुरंत तैनात हों। आपातकालीन प्रोटोकॉल सक्रिय।",
            "mr": "अतिगंभीर गर्दी दाब. सर्व सुरक्षा कर्मचाऱ्यांनी तात्काळ प्रतिसाद द्यावा. आपत्कालीन प्रोटोकॉल सक्रिय."
        },
        "attention security. a possible lost child has been detected. please check the flagged zone immediately.": {
            "hi": "सुरक्षा ध्यान दें। एक संभावित खोए हुए बच्चे का पता चला है। चिह्नित ज़ोन की तुरंत जांच करें।",
            "mr": "सुरक्षा लक्ष द्या. एक संभाव्य हरवलेले मूल आढळले आहे. कृपया चिन्हांकित झोनची त्वरित तपासणी करा."
        },
        "a dangerous crowd compression wave was detected propagating through the tracked area.": {
            "hi": "निगरानी क्षेत्र में एक खतरनाक भीड़ संपीड़न लहर फैलती हुई पाई गई।",
            "mr": "ट्रॅक केलेल्या परिसरात घातक गर्दी दाब लाट पसरताना आढळली."
        },
        "normal monitoring state.": {
            "hi": "सामान्य निगरानी स्थिति।",
            "mr": "सामान्य देखरेख स्थिती."
        },
        "no recent incident recorded.": {
            "hi": "हाल ही में कोई घटना दर्ज नहीं की गई।",
            "mr": "कोणतीही अलीकडील घटना नोंदवली नाही."
        },
        "safe": {
            "hi": "सुरक्षित",
            "mr": "सुरक्षित"
        },
        "warning": {
            "hi": "चेतावनी",
            "mr": "चेतावणी"
        },
        "danger": {
            "hi": "खतरा",
            "mr": "धोका"
        },
        "critical": {
            "hi": "गंभीर",
            "mr": "अतिगंभीर"
        }
    }

    # Dynamic parameterized patterns
    DYNAMIC_PATTERNS = [
        # Direction surge (X% aligned)
        (
            re.compile(r"Direction surge \((\d+)% aligned\)", re.IGNORECASE),
            "Direction surge ({0}% aligned)",
            {
                "hi": "दिशात्मक उछाल ({0}% संरेखित)",
                "mr": "दिशात्मक लाट ({0}% एका दिशेने)"
            }
        ),
        # Possible lost child(ren) detected: X
        (
            re.compile(r"Possible lost child\(ren\) detected:\s*(\d+)", re.IGNORECASE),
            "Possible lost child(ren) detected: {0}",
            {
                "hi": "संभावित खोए हुए बच्चे का पता चला: {0}",
                "mr": "संभाव्य हरवलेली मुले आढळली: {0}"
            }
        ),
        # Dangerous cluster compression developing: X dense clusters detected
        (
            re.compile(r"Dangerous cluster compression developing:\s*(\d+)\s*dense clusters detected", re.IGNORECASE),
            "Dangerous cluster compression developing: {0} dense clusters detected",
            {
                "hi": "खतरनाक क्लस्टर संपीड़न विकसित हो रहा है: {0} घने क्लस्टर मिले",
                "mr": "घातक क्लस्टर दाब वाढत आहे: {0} दाट क्लस्टर्स आढळले"
            }
        ),
        # Zone X intrusion: Y people
        (
            re.compile(r"Zone\s+([A-Za-z0-9_-]+)\s+intrusion:\s*(\d+)\s*people", re.IGNORECASE),
            "Zone {0} intrusion: {1} people",
            {
                "hi": "ज़ोन {0} में अनधिकृत प्रवेश: {1} लोग",
                "mr": "झोन {0} मध्ये अनधिकृत प्रवेश: {1} लोक"
            }
        ),
        # Speed spike detected: X px/f
        (
            re.compile(r"Speed spike detected:\s*([\d\.]+)\s*px/f", re.IGNORECASE),
            "Speed spike detected: {0} px/f",
            {
                "hi": "गति में तीव्र वृद्धि दर्ज: {0} px/f",
                "mr": "वेगात अचानक वाढ नोंदवली: {0} px/f"
            }
        ),
        # High crowd density: X%
        (
            re.compile(r"High crowd density:\s*([\d\.]+)%", re.IGNORECASE),
            "High crowd density: {0}%",
            {
                "hi": "उच्च भीड़ घनत्व: {0}%",
                "mr": "उच्च गर्दी घनता: {0}%"
            }
        ),
        # Acoustic panic detected: X dB
        (
            re.compile(r"Acoustic panic detected:\s*([\d\.]+)\s*dB", re.IGNORECASE),
            "Acoustic panic detected: {0} dB",
            {
                "hi": "ध्वनि पैनिक स्तर दर्ज: {0} dB",
                "mr": "आवाज पॅनिक पातळी नोंदवली: {0} dB"
            }
        ),
        # Alert driven by: X
        (
            re.compile(r"Alert driven by:\s*(.+)", re.IGNORECASE),
            "Alert driven by: {0}",
            {
                "hi": "अलर्ट का मुख्य कारण: {0}",
                "mr": "अलर्टचे मुख्य कारण: {0}"
            }
        )
    ]

    def __init__(self):
        self.gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")

    def _extract_tokens(self, text: str) -> Tuple[str, List[str]]:
        """
        Extract numerical values, percentages, timestamps, and classifications
        so they remain strictly unaltered.
        """
        tokens = []
        token_pattern = re.compile(
            r"(\b\d{1,2}:\d{2}(?::\d{2})?\b|\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?x|\b\d+(?:\.\d+)?\s*(?:dB|px/f|people|sec|s)\b|\b\d+(?:\.\d+)?\b|\bCRITICAL\b|\bDANGER\b|\bWARNING\b|\bSAFE\b|\bCAM-\d+\b|\bZone-[A-Za-z0-9_-]+\b)"
        )

        def repl(match):
            tokens.append(match.group(0))
            return f"__T{len(tokens) - 1}__"

        masked_text = token_pattern.sub(repl, text)
        return masked_text, tokens

    def _restore_tokens(self, text: str, tokens: List[str]) -> str:
        """Restore extracted numerical values and classifications exactly."""
        for idx, tok in enumerate(tokens):
            text = text.replace(f"__T{idx}__", tok)
        return text

    def translate_message(self, message: str, target_lang: str = "en") -> str:
        """
        Translate a single alert string into target_lang ('en', 'hi', 'mr').
        Guarantees numbers, timestamps, and levels are preserved.
        """
        if not message or target_lang == "en":
            return message

        target_lang = target_lang.lower()
        if target_lang not in ("hi", "mr"):
            return message

        # Clean leading icons (e.g. "⚠ ", "🔶 ", "🔴 ")
        prefix = ""
        icon_match = re.match(r"^([⚠️🔴🔶🟢🚨\s]+)", message)
        if icon_match:
            prefix = icon_match.group(1)
            core_msg = message[len(prefix):].strip()
        else:
            core_msg = message.strip()

        # 1. Check exact match in dictionary (case-insensitive)
        norm_key = core_msg.lower().strip()
        if norm_key in self.DICTIONARY and target_lang in self.DICTIONARY[norm_key]:
            return prefix + self.DICTIONARY[norm_key][target_lang]

        # 2. Check dynamic parameterized patterns
        for pattern, _, lang_templates in self.DYNAMIC_PATTERNS:
            m = pattern.search(core_msg)
            if m:
                groups = m.groups()
                template = lang_templates.get(target_lang)
                if template:
                    translated_part = template.format(*groups)
                    # Replace in original message to preserve surrounding words
                    replaced = pattern.sub(translated_part, core_msg)
                    return prefix + replaced

        # 3. Mask tokens (numbers, timestamps, classifications), translate phrase keywords, and restore
        masked_text, tokens = self._extract_tokens(core_msg)
        
        # Keyword substitutions for words inside sentences
        hi_words = {
            "Multiple metrics elevated": "कई पैरामीटर बढ़े हुए हैं",
            "Direction surge": "दिशात्मक उछाल",
            "aligned": "संरेखित",
            "people": "लोग",
            "dense clusters": "घने क्लस्टर",
            "compression": "संपीडन",
            "wave": "लहर",
            "detected": "पता चला",
            "alert": "अलर्ट",
            "intrusion": "अनधिकृत प्रवेश",
            "high": "उच्च",
            "speed": "गति",
            "density": "घनत्व",
            "risk": "जोखिम"
        }
        mr_words = {
            "Multiple metrics elevated": "अनेक मापदंड वाढलेले आहेत",
            "Direction surge": "दिशात्मक लाट",
            "aligned": "एका दिशेने",
            "people": "लोक",
            "dense clusters": "दाट क्लस्टर्स",
            "compression": "दाब",
            "wave": "लाट",
            "detected": "आढळले",
            "alert": "इशारा",
            "intrusion": "अनधिकृत प्रवेश",
            "high": "उच्च",
            "speed": "वेग",
            "density": "घनता",
            "risk": "धोका"
        }

        word_map = hi_words if target_lang == "hi" else mr_words
        translated = masked_text
        for eng, tr in word_map.items():
            pattern = re.compile(re.escape(eng), re.IGNORECASE)
            translated = pattern.sub(tr, translated)

        final_msg = self._restore_tokens(translated, tokens)
        return prefix + final_msg

    def translate_alert(self, alert_dict: Dict[str, Any], target_lang: str = "en") -> Dict[str, Any]:
        """
        Translate an alert dictionary (with message, time_str, level, metrics).
        Preserves all metadata, level, timestamp, metrics untouched.
        """
        if target_lang == "en":
            return dict(alert_dict)

        translated = dict(alert_dict)
        orig_msg = alert_dict.get("message", "")
        translated["original_message"] = orig_msg
        translated["message"] = self.translate_message(orig_msg, target_lang=target_lang)
        translated["lang"] = target_lang
        # Level remains untouched for classification integrity
        translated["level"] = alert_dict.get("level", "SAFE")
        return translated

    def translate_feed(self, alerts: List[Dict[str, Any]], target_lang: str = "en") -> List[Dict[str, Any]]:
        """Translate a batch of alerts."""
        return [self.translate_alert(a, target_lang=target_lang) for a in alerts]

