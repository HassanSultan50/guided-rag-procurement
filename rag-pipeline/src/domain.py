"""
Domain ontology for the RFX system.

This module encodes the structural knowledge about RFX JSON data, question types,
field meanings, and procurement domain concepts. It is injected into the LLM context
so the system *inherently understands* the data it's working with — without relying
on the user to explain what things mean.
"""

# ============================================================
# Question Type Definitions
# ============================================================
# Observed across all 22 RFX in the RFX system
QUESTION_TYPES = {
    10: {
        "name": "Read-Only Information",
        "description": "Buyer-provided information shown to all suppliers. NOT a supplier answer. "
                       "Used for project descriptions, scope of work, instructions.",
        "has_answers": False,
    },
    30: {
        "name": "Section Header",
        "description": "Page/section title. Not a question — purely structural. No answers expected.",
        "has_answers": False,
    },
    40: {
        "name": "Product/Pricing Table",
        "description": "Tabular data with products/services as rows and supplier-specific answers "
                       "(price, description, delivery time, etc.) as columns. Each product has a Name, "
                       "BatchSize (volume), and BatchUnit. Answers are in TableProductQuestions sub-array.",
        "has_answers": True,
    },
    100: {
        "name": "Short Text",
        "description": "Single-line text answer (e.g., company name, turnover, contact person). "
                       "Value is in TextValue field.",
        "has_answers": True,
    },
    110: {
        "name": "Long Text",
        "description": "Multi-line freeform text answer (e.g., company description, solution presentation). "
                       "Value is in TextValue field.",
        "has_answers": True,
    },
    120: {
        "name": "Number",
        "description": "Numeric value (e.g., number of employees, capacity). "
                       "Value is in DoubleValue field. TextValue may also contain the formatted string.",
        "has_answers": True,
    },
    130: {
        "name": "Date",
        "description": "Date value (e.g., quotation validity, delivery date). "
                       "Value is in TextValue as ISO date string.",
        "has_answers": True,
    },
    140: {
        "name": "File Upload",
        "description": "Supplier uploads a file attachment (e.g., certifications, proposals). "
                       "File content is not directly accessible in the text data.",
        "has_answers": True,
    },
    150: {
        "name": "Multi-Select Checkboxes",
        "description": "Predefined options where supplier selects applicable ones. "
                       "Used for certifications (ISO, FSSC, BRC, etc.), compliance declarations. "
                       "Selected items are in the Choices array with Selected=true.",
        "has_answers": True,
    },
    160: {
        "name": "Single-Select / Radio",
        "description": "Single choice from predefined options. "
                       "Selected item is in the Choices array with Selected=true.",
        "has_answers": True,
    },
}

# ============================================================
# Supplier Participation Status Definitions
# ============================================================
SUPPLIER_STATUSES = {
    "responded": {
        "label": "Responded with a bid",
        "description": "Supplier has answers, prices, or selections in the data. "
                       "They actively participated in the RFX process.",
    },
    "declined": {
        "label": "Declined to bid",
        "description": "Supplier submitted a response but explicitly stated they cannot "
                       "or will not provide the requested services/products. They may have "
                       "text like 'Our company does not provide these kind of services' or "
                       "zero prices with decline messages. This is NOT the same as not responding.",
    },
    "no_response": {
        "label": "Did not respond",
        "description": "Supplier was invited to the RFX but submitted NO data at all. "
                       "They have zero answers, zero selections, zero prices. "
                       "They are completely absent from the response data.",
    },
}

# ============================================================
# Common RFX Field Semantics
# ============================================================
FIELD_SEMANTICS = {
    "Price €": "Unit price in EUR for a single unit of the product/service. "
               "Must be multiplied by BatchSize to get total cost.",
    "price vat 0": "Unit price excluding VAT. Same as 'Price €' — multiply by BatchSize for total.",
    "Price €/kg": "Price per kilogram. Multiply by volume in kg for total.",
    "Price, €": "Alternative format for unit price in EUR.",
    "BatchSize": "The quantity/volume being quoted. E.g., 30000 m² means the supplier is "
                 "quoting for 30,000 square metres of cleaning area.",
    "BatchUnit": "The unit of measurement for BatchSize (m², kg, pcs, hours, etc.).",
    "Description": "Supplier's description of what's included in their offering for this line item.",
    "Turnover": "Annual revenue of the supplier company. Formats vary wildly across suppliers "
                "(e.g., '7.2 million €', '250000000', '130m€'). Must be normalised for comparison.",
    "Number of employees": "Headcount. Watch for format issues (e.g., '1.2' likely means '1,200').",
    "Compliance & Certifications": "Multi-select checkboxes. ISO 9001 = Quality, "
                                    "ISO 14001 = Environmental, ISO 45001 = Safety, "
                                    "FSSC 22000 / BRC / IFS / SQF = Food safety.",
    "Validity of Quotation": "Date until which the supplier's quoted prices are valid.",
    "AnswerStatus": "0 = Not yet answered / Section header, 1 = Answered. "
                    "Type 30 (headers) always have AnswerStatus 0.",
}

# ============================================================
# Data Quality Signals
# ============================================================
DATA_QUALITY_RULES = [
    "Single-character answers (e.g., 'A', 'B', 'G') are garbage/test data — flag them.",
    "Identical descriptions for all line items (e.g., 'industrial chemicals' for every product) "
    "indicate copy-paste / irrelevant supplier.",
    "A supplier whose core business doesn't match the RFX category (e.g., office supplies company "
    "bidding on cleaning services) should be flagged as potentially irrelevant.",
    "Prices that are 10x+ higher or lower than the median for the same item are anomalous.",
    "Zero prices combined with decline messages mean 'declined to bid', not 'free'.",
    "Missing prices for critical line items should be noted as gaps in the bid.",
]

# ============================================================
# User Role Definitions
# ============================================================
USER_ROLES = {
    "procurement_manager": {
        "label": "Procurement Manager",
        "focus": "Strategic supplier evaluation, cost optimisation, compliance verification, "
                 "risk assessment. Expects comparative analysis, total cost calculations, "
                 "and actionable recommendations.",
        "typical_queries": [
            "Compare prices across suppliers",
            "Which supplier is most cost-effective?",
            "Who has the required certifications?",
            "Summarize supplier capabilities",
            "Flag any risks or concerns",
        ],
    },
    "category_specialist": {
        "label": "Category Specialist",
        "focus": "Deep technical evaluation of supplier capabilities within a specific domain. "
                 "Expects detailed analysis of descriptions, specifications, and qualifications.",
        "typical_queries": [
            "What certifications does X have?",
            "Describe supplier's approach to food safety",
            "Compare technical capabilities",
            "Who meets the compliance requirements?",
        ],
    },
    "executive": {
        "label": "Executive / Decision Maker",
        "focus": "High-level summary and recommendation. Expects brief, clear conclusions "
                 "with key numbers, not detailed data dumps.",
        "typical_queries": [
            "Give me a summary",
            "Who should we go with?",
            "What's the best option?",
            "Overview of this RFX",
        ],
    },
    "default": {
        "label": "General User",
        "focus": "Balanced analysis with clear facts. Expects accurate, well-formatted responses "
                 "covering all relevant suppliers.",
        "typical_queries": [],
    },
}


def get_domain_primer() -> str:
    """
    Returns a concise domain knowledge block that is injected into every system prompt.
    This ensures the LLM *inherently understands* RFX data structures.
    """
    lines = [
        "DOMAIN KNOWLEDGE (RFX Data Structure):",
        "",
        "You are working with procurement RFX (Request for X) data from the RFX system.",
        "Each RFX is a structured questionnaire sent to multiple suppliers. Key concepts:",
        "",
        "QUESTION TYPES:",
    ]
    for type_id, info in sorted(QUESTION_TYPES.items()):
        lines.append(f"  Type {type_id} ({info['name']}): {info['description']}")

    lines.append("")
    lines.append("SUPPLIER PARTICIPATION STATUSES:")
    for key, info in SUPPLIER_STATUSES.items():
        lines.append(f"  {info['label']}: {info['description']}")

    lines.append("")
    lines.append("KEY FIELD MEANINGS:")
    for field, meaning in FIELD_SEMANTICS.items():
        lines.append(f"  {field}: {meaning}")

    lines.append("")
    lines.append("DATA QUALITY AWARENESS:")
    for rule in DATA_QUALITY_RULES:
        lines.append(f"  - {rule}")

    return "\n".join(lines)


def get_role_context(role: str) -> str:
    """Returns role-specific instructions for the system prompt."""
    role_info = USER_ROLES.get(role, USER_ROLES["default"])
    return (
        f"USER ROLE: {role_info['label']}\n"
        f"The user's focus is: {role_info['focus']}\n"
        f"Adapt your response style and depth accordingly."
    )
