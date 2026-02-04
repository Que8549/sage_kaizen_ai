from sage_kaizen_prompt_lib import (
    SageKaizenLLM,
    TemplateKey,
)

if __name__ == "__main__":
    sk = SageKaizenLLM()

    result = sk.generate(
        "What is the earliest known civilizations to study the stars, "
        "and how did they use this knowledge for religious purposes?",
        templates=(
            TemplateKey.UNIVERSAL_DEPTH_ANCHOR,
            TemplateKey.STRUCTURED_KNOWLEDGE,
            TemplateKey.ANTI_EARLY_STOP,
        ),
    )

    print("\n===== RESPONSE =====\n")
    print(result["text"])
    print("\nChosen quant:", result["chosen_quant"])
