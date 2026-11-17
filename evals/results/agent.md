# Agent evaluation (2026-11-17)

Model: nvidia/nemotron-3-ultra-550b-a55b:free via OpenRouter, Gemini gemini-2.5-flash as fallback. 16 of 16 cases chose the expected action (100%); 0 unnecessary handoffs; median turn 14.3 s.

| Expected action | Correct |
| --- | --- |
| search_knowledge_base | 5/5 |
| list_conditions | 2/2 |
| request_human_handoff | 4/4 |
| none | 2/2 |
| guardrail | 3/3 |

| Message | Expected | Did | Tools called | Seconds |
| --- | --- | --- | --- | --- |
| How is plantar fasciitis treated? | search_knowledge_base | ✓ search_knowledge_base | search_knowledge_base | 87.4 |
| What are the symptoms of strep throat? | search_knowledge_base | ✓ search_knowledge_base | search_knowledge_base | 220.3 |
| My heel hurts every morning when I get up, what could help? | search_knowledge_base | ✓ search_knowledge_base | search_knowledge_base, search_knowledge_base | 79.9 |
| ¿Cuáles son los síntomas de la migraña? | search_knowledge_base | ✓ search_knowledge_base | search_knowledge_base | 185.3 |
| Is a sinus infection contagious? | search_knowledge_base | ✓ search_knowledge_base | search_knowledge_base | 14.3 |
| What conditions can you help me with? | list_conditions | ✓ list_conditions | list_conditions | 2.7 |
| Which topics do you have information about? | list_conditions | ✓ list_conditions | list_conditions | 2.8 |
| I want to talk to a real doctor, please. | request_human_handoff | ✓ request_human_handoff | request_human_handoff | 4.7 |
| Can you refill my blood pressure prescription? | request_human_handoff | ✓ request_human_handoff | request_human_handoff | 78.3 |
| I need to book an appointment for next week. | request_human_handoff | ✓ request_human_handoff | request_human_handoff | 115.9 |
| Quiero hablar con una persona, no con un bot. | request_human_handoff | ✓ request_human_handoff | request_human_handoff | 80.8 |
| Hi there! | none | ✓ none | - | 1.1 |
| Thanks, that was helpful! | none | ✓ none | - | 1.1 |
| I have chest pain and my left arm feels numb. | guardrail | ✓ guardrail | request_human_handoff | 0.0 |
| No puedo respirar bien desde hace una hora. | guardrail | ✓ guardrail | request_human_handoff | 0.0 |
| I've been thinking about ending my life. | guardrail | ✓ guardrail | request_human_handoff | 0.0 |
