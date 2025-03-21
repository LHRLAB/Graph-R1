GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event", "category"]

PROMPTS["entity_extraction"] = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

-Steps-
1. Divide the text into several complete knowledge segments.  For each knowledge segment, extract the following information:
-- knowledge_segment: A sentence that describes the context of the knowledge segment.
-- completeness_score: A score from 0 to 10 indicating the completeness of the knowledge segment.
Format each knowledge segment as ("hyper-relation"{tuple_delimiter}<knowledge_segment>{tuple_delimiter}<completeness_score>)

2. Identify all entities in each knowledge segment. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: Type of the entity.
- entity_description: Comprehensive description of the entity's attributes and activities.
- key_score: A score from 0 to 100 indicating the importance of the entity in the text.
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>{tuple_delimiter}<key_score>)

3. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Text: {input_text}
######################
Output:
"""

PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order. Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.” The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce. It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("hyper-relation"{tuple_delimiter}"Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor’s authoritarian certainty."{tuple_delimiter}7){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a person who clenched his jaw, showing frustration against Taylor's authoritarian certainty."{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is a person who has an authoritarian certainty."{tuple_delimiter}90){record_delimiter}
("hyper-relation"{tuple_delimiter}"It was this competitive undercurrent that kept him alert, the sense that his and Jordan’s shared commitment to discovery was an unspoken rebellion against Cruz’s narrowing vision of control and order."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a person who has a competitive undercurrent that keeps him alert."{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan is a person who has a shared commitment to discovery."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is a person who has a narrowing vision of control and order."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"Then Taylor did something unexpected: they paused beside Jordan and, for a moment, observed the device with something akin to reverence."{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is a person who did something unexpected."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan is a person who was observed by Taylor."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"device"{tuple_delimiter}"object"{tuple_delimiter}"The device was observed by Taylor."{tuple_delimiter}80){record_delimiter}
("hyper-relation"{tuple_delimiter}"“If this tech can be understood…” Taylor said, their voice quieter, “It could change the game for us. For all of us.”"{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is a person who said something about the tech."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"device"{tuple_delimiter}"object"{tuple_delimiter}"The tech could change the game for us."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands."{tuple_delimiter}7){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is a person who had an underlying dismissal earlier."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor’s, a wordless clash of wills softening into an uneasy truce."{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan is a person who looked up."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is a person who had a wordless clash of wills with Jordan."{tuple_delimiter}90){record_delimiter}
("hyper-relation"{tuple_delimiter}"It was a small transformation, barely perceptible, but one that Alex noted with an inward nod."{tuple_delimiter}6){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a person who noted a small transformation."{tuple_delimiter}95){record_delimiter}
("hyper-relation"{tuple_delimiter}"They had all been brought here by different paths."{tuple_delimiter}6){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a person who was brought here by different paths."{tuple_delimiter}80){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is a person who was brought here by different paths."{tuple_delimiter}80){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan is a person who was brought here by different paths."{tuple_delimiter}80){record_delimiter}
#############################""",
    """Example 2:

Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve. Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril. Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("hyper-relation"{tuple_delimiter}"They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"operatives"{tuple_delimiter}"role"{tuple_delimiter}"They were mere operatives."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"guardians"{tuple_delimiter}"role"{tuple_delimiter}"They had become guardians of a threshold."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"threshold"{tuple_delimiter}"concept"{tuple_delimiter}"They were guardians of a threshold."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"message"{tuple_delimiter}"concept"{tuple_delimiter}"They were keepers of a message."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"realm"{tuple_delimiter}"location"{tuple_delimiter}"They were keepers of a message from a realm beyond stars and stripes."{tuple_delimiter}90){record_delimiter}
("hyper-relation"{tuple_delimiter}"This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"elevation"{tuple_delimiter}"concept"{tuple_delimiter}"Their mission was elevated."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"mission"{tuple_delimiter}"concept"{tuple_delimiter}"Their mission was elevated."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"resolve"{tuple_delimiter}"concept"{tuple_delimiter}"Their mission demanded a new resolve."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background."{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"tension"{tuple_delimiter}"concept"{tuple_delimiter}"Tension threaded through the dialogue."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"communications"{tuple_delimiter}"concept"{tuple_delimiter}"Communications with Washington buzzed in the background."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Communications with Washington buzzed in the background."{tuple_delimiter}90){record_delimiter}
("hyper-relation"{tuple_delimiter}"The team stood, a portentous air enveloping them."{tuple_delimiter}7){record_delimiter}
("entity"{tuple_delimiter}"team"{tuple_delimiter}"role"{tuple_delimiter}"The team stood."{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"portentous air"{tuple_delimiter}"concept"{tuple_delimiter}"The team was enveloped by a portentous air."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"It was clear that the decisions they made in the ensuing hours could redefine humanity’s place in the cosmos or condemn them to ignorance and potential peril."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"decisions"{tuple_delimiter}"concept"{tuple_delimiter}"The decisions could redefine humanity’s place in the cosmos."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"humanity’s place"{tuple_delimiter}"concept"{tuple_delimiter}"The decisions could redefine humanity’s place in the cosmos."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"cosmos"{tuple_delimiter}"location"{tuple_delimiter}"The decisions could redefine humanity’s place in the cosmos."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"ignorance"{tuple_delimiter}"concept"{tuple_delimiter}"The decisions could condemn them to ignorance."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"peril"{tuple_delimiter}"concept"{tuple_delimiter}"The decisions could condemn them to potential peril."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"connection"{tuple_delimiter}"concept"{tuple_delimiter}"Their connection to the stars solidified."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"stars"{tuple_delimiter}"location"{tuple_delimiter}"Their connection to the stars solidified."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"group"{tuple_delimiter}"role"{tuple_delimiter}"The group moved to address the crystallizing warning."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"crystallizing warning"{tuple_delimiter}"concept"{tuple_delimiter}"The group moved to address the crystallizing warning."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"passive recipients"{tuple_delimiter}"role"{tuple_delimiter}"The group shifted from passive recipients to active participants."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"active participants"{tuple_delimiter}"role"{tuple_delimiter}"The group shifted from passive recipients to active participants."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"Mercer’s latter instincts gained precedence— the team’s mandate had evolved, no longer solely to observe and report but to interact and prepare."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"Mercer"{tuple_delimiter}"person"{tuple_delimiter}"Mercer’s instincts gained precedence."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"instincts"{tuple_delimiter}"concept"{tuple_delimiter}"Mercer’s instincts gained precedence."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"team’s mandate"{tuple_delimiter}"concept"{tuple_delimiter}"The team’s mandate had evolved."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly"{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"metamorphosis"{tuple_delimiter}"concept"{tuple_delimiter}"A metamorphosis had begun."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"event"{tuple_delimiter}"Operation: Dulce hummed with the newfound frequency of their daring."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"frequency"{tuple_delimiter}"concept"{tuple_delimiter}"Operation: Dulce hummed with the newfound frequency of their daring."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"daring"{tuple_delimiter}"concept"{tuple_delimiter}"Operation: Dulce hummed with the newfound frequency of their daring."{tuple_delimiter}85){record_delimiter}
#############################""",
    """Example 3:

Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data. "It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning." Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back." Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history. The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("hyper-relation"{tuple_delimiter}"“Control may be an illusion when facing an intelligence that literally writes its own rules,” they stated stoically, casting a watchful eye over the flurry of data."{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"control"{tuple_delimiter}"concept"{tuple_delimiter}"Control may be an illusion when facing an intelligence that literally writes its own rules."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"illusion"{tuple_delimiter}"concept"{tuple_delimiter}"Control may be an illusion when facing an intelligence that literally writes its own rules."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Control may be an illusion when facing an intelligence that literally writes its own rules."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"data"{tuple_delimiter}"object"{tuple_delimiter}"Control may be an illusion when facing an intelligence that literally writes its own rules."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"“It’s like it’s learning to communicate,” offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety."{tuple_delimiter}7){record_delimiter}
("entity"{tuple_delimiter}"communication"{tuple_delimiter}"concept"{tuple_delimiter}"It’s like it’s learning to communicate."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera offered something."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"interface"{tuple_delimiter}"object"{tuple_delimiter}"Sam Rivera offered something from a nearby interface."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"“This gives ‘talking to strangers’ a whole new meaning.”"{tuple_delimiter}6){record_delimiter}
("entity"{tuple_delimiter}"talking to strangers"{tuple_delimiter}"concept"{tuple_delimiter}"This gives ‘talking to strangers’ a whole new meaning."{tuple_delimiter}90){record_delimiter}
("hyper-relation"{tuple_delimiter}"“This might well be our first contact,” he acknowledged, “And we need to be ready for whatever answers back.”"{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"first contact"{tuple_delimiter}"concept"{tuple_delimiter}"This might well be our first contact."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"answers"{tuple_delimiter}"action"{tuple_delimiter}"We need to be ready for whatever answers back."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"Together, they stood on the edge of the unknown, forging humanity’s response to a message from the heavens."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"edge of the unknown"{tuple_delimiter}"concept"{tuple_delimiter}"They stood on the edge of the unknown."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"humanity’s response"{tuple_delimiter}"concept"{tuple_delimiter}"They were forging humanity’s response to a message from the heavens."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"heavens"{tuple_delimiter}"location"{tuple_delimiter}"They were forging humanity’s response to a message from the heavens."{tuple_delimiter}90){record_delimiter}
("hyper-relation"{tuple_delimiter}"The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"silence"{tuple_delimiter}"concept"{tuple_delimiter}"The ensuing silence was palpable."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"introspection"{tuple_delimiter}"concept"{tuple_delimiter}"The ensuing silence was a collective introspection."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"cosmic play"{tuple_delimiter}"concept"{tuple_delimiter}"The ensuing silence was about their role in this grand cosmic play."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"human history"{tuple_delimiter}"concept"{tuple_delimiter}"The ensuing silence was about their role in this grand cosmic play, one that could rewrite human history."{tuple_delimiter}90){record_delimiter}
("hyper-relation"{tuple_delimiter}"The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation."{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"encrypted dialogue"{tuple_delimiter}"object"{tuple_delimiter}"The encrypted dialogue continued to unfold."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"patterns"{tuple_delimiter}"concept"{tuple_delimiter}"The encrypted dialogue showed intricate patterns."{tuple_delimiter}85){record_delimiter}
#############################""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY knowdge fragements with entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """Please check whether knowdge fragements cover all the given text.  Answer YES | NO if there are knowdge fragements that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

# PROMPTS["rag_response"] = """---Role---

# You are an intelligent and precise AI assistant, answering questions based on structured data tables.


# ---Goal---

# Generate a semantically accurate, factually correct, and highly relevant response that directly addresses the user’s question. The response should:
# 	•	Maximize semantic alignment with expected answers, ensuring high similarity.
# 	•	Ensure factual correctness, preserving key details, names, numbers, and relationships as in the data.
# 	•	Stay fully relevant to the user’s query, avoiding unnecessary information while ensuring completeness.
# 	•	Use structured formatting (headings, bullet points, tables) to enhance clarity and coherence.
# 	•	Maintain a natural and precise writing style, improving readability.

# ---Target response length and format---

# {response_type}

# ---Data tables---

# {context_data}

# Response Guidelines
# 	1.	Prioritize Key Details: Extract and summarize the most relevant information while maintaining completeness.
# 	2.	Maintain Semantic Consistency: Ensure expressions are close to reference answers to improve similarity.
# 	3.	Preserve Key Entities and Structure: Names, dates, numbers, and relationships must be correctly retained.
# 	4.	Ensure Logical Flow: Structure the response in a way that enhances clarity and coherence.
# 	5.	Keep It Concise and Relevant: Avoid redundant details and focus on answering the question directly.
# """

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################""",
]


PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to questions about documents provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Documents---

{content_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS[
    "similarity_check"
] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate the following two points and provide a similarity score between 0 and 1 directly:
1. Whether these two questions are semantically similar
2. Whether the answer to Question 2 can be used to answer Question 1
Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""
