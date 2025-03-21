from hypergraphrag import HyperGraphRAG
import os
os.environ["OPENAI_API_KEY"] = open("openai_api_key.txt").read().strip()

rag = HyperGraphRAG(working_dir=f"expr/hypertension")
query_text = 'How can the increasing prevalence of hypertension and elevated blood pressure, combined with the challenges of managing these conditions, inform the development of effective non-pharmacological interventions that promote lifestyle changes among patients, considering the socio-economic factors and patient-related barriers to adherence?'
result = rag.query(query_text)
print(result)



