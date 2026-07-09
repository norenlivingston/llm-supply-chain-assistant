"""Session 2 - reusable prompt templates."""
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    template: str

    def render(self, **kwargs) -> str:
        return self.template.format(**kwargs)


SUPPLIER_RISK_TEMPLATE = PromptTemplate(
    template=(
        "Assess supply risk for supplier '{supplier}' given the following "
        "context:\n{context}\n\n"
        "Return a risk level (low/medium/high) and a one-sentence rationale."
    )
)

ANOMALY_EXPLAIN_TEMPLATE = PromptTemplate(
    template=(
        "A metric named '{metric}' moved from {old_value} to {new_value} "
        "between {old_period} and {new_period}. Explain plausible supply "
        "chain causes in 2-3 bullet points."
    )
)

if __name__ == "__main__":
    print(SUPPLIER_RISK_TEMPLATE.render(
        supplier="Acme Cold Storage",
        context="Two late deliveries in the last 30 days; no prior incidents.",
    ))
