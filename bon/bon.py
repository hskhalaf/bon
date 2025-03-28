import random
import pandas as pd
import torch
import csv
# from huit_generate import annotate_prompts
from vllm import LLM, SamplingParams

model = LLM(model="meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size=2, max_model_len = 4096)

def load_prompts(num_prompts=5):
    prompts = [
        "What is the process of photosynthesis and why it is essential for life on Earth?",
        "What are the major causes and consequences of climate change in the 21st century?",
        "How does the human brain process language, and what are the challenges in developing AI that can do the same?",
        "What are the key differences between classical physics and quantum mechanics?",
        "What is the history and impact of the Industrial Revolution on modern society?",
        "How do vaccines work, and why are they important for public health?",
        "What are the major events that led to the fall of the Roman Empire?",
        "What are black holes, and how do they form in the universe?",
        "What are the fundamental principles of democracy and why are they important?",
        "How does machine learning work, and what are some of its practical applications?",
        "What is the process of natural selection and its role in evolution?",
        "What are the biggest challenges facing space exploration today?",
        "How do electric vehicles work, and what are their advantages over traditional gasoline-powered cars?",
        "What is the economic concept of supply and demand and how does it affect prices?",
        "What are the ethical concerns surrounding artificial intelligence and automation?",
        "What is the impact of globalization on different economies and cultures?",
        "How does the immune system defend the body against infections?",
        "What are the psychological effects of social media on young people?",
        "What is the structure and function of DNA in living organisms?",
        "How does the stock market work, and what factors influence stock prices?",
        "Describe the life cycle of a star, from its formation to its eventual fate.",
        "Explain the different layers of Earth's atmosphere and their respective roles.",
        "What are the potential benefits and risks of using CRISPR-Cas9 gene editing technology?",
        "How does the internet work, from the initial request in a web browser to the display of a webpage?  Include details on protocols, servers, and data transmission.",
        "Describe the process of cellular respiration and its importance in providing energy for living organisms.",
        "Explain the concept of plate tectonics and how it has shaped the Earth's surface over millions of years.",
        "What are the different types of renewable energy sources, and what are their respective advantages and disadvantages?",
        "How does a nuclear reactor generate electricity, and what are the safety concerns associated with nuclear power?",
        "Describe the various stages of the human sleep cycle and the functions they serve.",
        "Explain the concept of entropy and its implications for the universe.",
        "Detail the various methods used in modern astronomy to study distant galaxies and celestial objects.",
        "How do different types of telescopes (optical, radio, X-ray, etc.) work, and what unique information does each provide?",
        "What are the current leading theories about the origin and evolution of the universe (beyond just the Big Bang)?",
        "Explain the different types of chemical bonds and how they influence the properties of molecules.",
        "How does 3D printing work, and what are some of its current and potential applications?",
        "Describe the process of protein synthesis in cells, from DNA transcription to translation.",
        "What are the major challenges in developing sustainable and efficient fusion power?",
        "Explain the concept of quantum entanglement and its potential implications for future technologies.",
        "How do modern weather forecasting models work, and what are their limitations?",
        "Describe the process of fossilization and how it provides evidence for evolution.",
        "What is dark matter and dark energy, and what evidence supports their existence?",
        "Explain the different types of computer programming languages and their typical uses.",
        "How does blockchain technology work, and what are its potential applications beyond cryptocurrencies?",
        "What are the major components of a computer, and how do they interact with each other?",
        "Describe the different stages of drug development, from initial research to clinical trials and approval.",
        "Trace the major events and causes of World War I, and explain its long-term consequences.",
        "What were the key factors that contributed to the rise and fall of the Mongol Empire?",
        "Describe the major philosophical ideas of the Enlightenment and their impact on Western society.",
        "Analyze the causes and consequences of the American Civil Rights Movement.",
        "Explain the different forms of government (e.g., monarchy, democracy, oligarchy) and their relative strengths and weaknesses.",
        "What were the major cultural and intellectual achievements of the Renaissance?",
        "Trace the history of the Cold War, from its origins to its eventual end.",
        "Analyze the economic and social impact of the Great Depression.",
        "Describe the major religious and philosophical traditions of ancient Greece.",
        "What were the key factors that led to the Age of Exploration and European colonialism?",
        "Explain the concept of nationalism and its role in shaping modern history.",
        "Analyze the causes and consequences of the French Revolution.",
        "Describe the major social and political reforms of the Progressive Era in the United States.",
        "What were the long-term effects of the transatlantic slave trade on Africa, Europe, and the Americas?",
        "Explain the different theories of international relations (e.g., realism, liberalism, constructivism).",
        "Analyze the causes and consequences of the Russian Revolution.",
        "Describe the major economic systems (e.g., capitalism, socialism, communism) and their respective characteristics.",
        "What were the major achievements and failures of the United Nations?",
        "Explain the concept of globalization and its impact on different cultures and economies.",
        "Analyze the causes and consequences of the Arab Spring uprisings.",
        "Describe the major historical developments that led to the formation of the European Union.",
        "What were the key factors that contributed to the rise of ancient civilizations in Mesopotamia, Egypt, and the Indus Valley?",
        "Explain the different schools of thought in psychology (e.g., behaviorism, psychoanalysis, cognitive psychology).",
        "Analyze the causes and consequences of the Rwandan genocide.",
        "Describe the major social and economic changes that occurred during the Neolithic Revolution.",
        "Analyze the major themes and stylistic features of a specific art movement (e.g., Impressionism, Surrealism, Cubism).",
        "Describe the evolution of a particular musical genre (e.g., jazz, rock, classical) over time.",
        "Explain the key elements of storytelling in film, including narrative structure, character development, and cinematography.",
        "Analyze the social and cultural significance of a particular work of literature (e.g., a novel, play, or poem).",
        "Describe the different forms of theatrical performance (e.g., tragedy, comedy, musical theatre).",
        "What are the key principles of design, and how are they applied in different fields (e.g., graphic design, architecture, fashion)?",
        "Analyze the role of music in a particular culture or society.",
        "Describe the different types of dance and their cultural significance.",
        "Explain the concept of 'art for art's sake' and its historical context.",
        "Analyze the impact of technology on the creation and consumption of art.",
        "Describe the history and evolution of photography as an art form.",
        "What are the major elements of a compelling narrative, and how do they contribute to a story's impact?",
        "Explain the different types of architectural styles and their historical and cultural influences.",
        "Analyze the use of symbolism and metaphor in a particular work of art or literature.",
        "Describe the role of museums and galleries in preserving and promoting art.",
        "Explain the different ethical theories (e.g., utilitarianism, deontology, virtue ethics) and how they might be applied to a specific moral dilemma.",
        "What is the philosophical concept of free will, and what are the arguments for and against its existence?",
        "Describe the major schools of thought in epistemology (the study of knowledge).",
        "What is the mind-body problem, and what are the different philosophical approaches to it?",
        "Explain the concept of justice and its different interpretations (e.g., distributive justice, retributive justice).",
        "What is the nature of consciousness, and what are the challenges in understanding it?",
        "Describe the different philosophical approaches to the meaning of life.",
        "What are the ethical implications of artificial intelligence and advanced robotics?",
        "Explain the concept of existentialism and its key ideas.",
        "What is the relationship between morality and religion?",
        "Imagine you could travel anywhere in time and space. Where and when would you go, and what would you hope to learn or experience?",
        "If you could design a perfect society, what would its core values and principles be, and how would it function?",
        "Describe a plausible scenario for first contact with an extraterrestrial civilization, and what the potential consequences might be.",
        "If you could have any superpower, what would it be, and how would you use it to make the world a better place (or a worse place, if you prefer)?",
        "Create a detailed description of a fictional world, including its geography, culture, history, and inhabitants.",
        "Imagine you are the leader of a newly independent nation. What are your top priorities, and how would you address the challenges facing your country?",
        "If you could invent a new technology that would solve a major global problem, what would it be, and how would it work?",
        "Describe a scenario in which artificial intelligence surpasses human intelligence, and what the implications might be for humanity.",
        "If you could have a conversation with any historical figure, who would it be, and what would you ask them?",
        "Imagine you are stranded on a desert island. What skills and knowledge would be most valuable for your survival, and how would you attempt to escape?",
        "If you could create a new form of art, what would it be, and how would it differ from existing art forms?",
        "Imagine you have the ability to communicate with animals. What would you learn from them, and how would it change your perspective on the world?",
        "If you could redesign the human body, what changes would you make, and why?",
        "Describe a utopian future for humanity, and what steps would be necessary to achieve it.",
        "Imagine you could live in any fictional universe (from a book, movie, or game). Which one would you choose, and why?",
        "If you could change one major event in history, what would it be, and what would the potential consequences be?",
        "Describe a dystopian future for humanity, and how might society have arrived at that state?",
        "If you could instantly learn any skill or language, which would you choose, and how would you use it?",
        "Imagine you are tasked with creating a new educational system. What would be its core principles, and how would it differ from current systems?",
        "Describe a plausible scenario for the collapse of a major civilization, and what lessons could be learned from it.",
        "If you could bring back one extinct species, which would it be, and what would be the potential ecological consequences?",
        "Imagine you are a detective investigating a complex crime. Describe the scene, the evidence, and your thought process as you try to solve the case.",
        "If you could create a new holiday, what would it celebrate, and what traditions would be associated with it?",
        "Describe a future where humans have colonized Mars. What are the challenges, opportunities, and societal structures of this new colony?",
        "If you could live forever, would you want to? Explain the potential benefits and drawbacks of immortality.",
        "Imagine you have the power to reshape the Earth's geography",
        "How does a compiler translate high-level programming code into machine-executable instructions?",
        "Describe the different types of network topologies (e.g., bus, star, ring, mesh) and their advantages and disadvantages.",
        "What are the major challenges in developing effective treatments for neurodegenerative diseases like Alzheimer's and Parkinson's?",
        "Explain the different types of data structures (e.g., arrays, linked lists, trees, graphs) and their use cases.",
        "How does a search engine like Google work, from crawling and indexing web pages to ranking search results?",
        "Describe the process of meiosis and its significance in sexual reproduction.",
        "What are the major ethical considerations in conducting clinical trials on human subjects?",
        "Explain the different types of machine learning algorithms (e.g., supervised, unsupervised, reinforcement learning).",
        "How does a virtual private network (VPN) work, and what are its security benefits?",
        "Describe the different types of cloud computing services (e.g., IaaS, PaaS, SaaS) and their use cases.",
        "What are the major challenges in developing self-driving cars, both technically and ethically?",
        "Explain the concept of object-oriented programming and its key principles (e.g., encapsulation, inheritance, polymorphism).",
        "How does a database management system (DBMS) store and manage data efficiently?",
        "Describe the different types of cyberattacks (e.g., phishing, malware, denial-of-service) and their potential impact.",
        "What are the major challenges in developing sustainable agriculture practices that can feed a growing global population?",
        "Explain the different phases of the software development lifecycle (SDLC).",
        "How does a digital camera capture and process images?",
        "Describe the different types of memory in a computer system (e.g., RAM, ROM, cache, hard drive) and their functions.",
        "What are the major challenges in developing effective treatments for cancer?",
        "Explain the concept of Big Data and the technologies used to analyze it.",
        "How does GPS (Global Positioning System) work?",
        "Describe the different stages of a project management lifecycle.",
        "What are the major challenges in developing quantum computers?",
        "Explain the concept of recursion in computer programming.",
        "How does a web server handle multiple client requests concurrently?",
        "What are the key principles of effective user interface (UI) and user experience (UX) design?",
        "How does a cryptocurrency like Bitcoin work, including the concepts of mining, blockchain, and transactions?",
        "What are the key features and functions of an operating system?",
        "How do recommendation systems (like those used by Netflix or Amazon) work?",
        "Describe different techniques for data compression.",
        "What are the challenges in developing effective strategies for cybersecurity?",
        "How does image recognition software work?",
        "Describe the different methods for testing software.",
        "What are the potential impacts of automation and artificial intelligence on the job market?",
        "Explain the different approaches to natural language processing (NLP).",
        "How does a wireless network (e.g., Wi-Fi) operate?",
        "What are the major challenges in developing effective treatments for viral infections?",
        "How does a search algorithm (e.g., binary search, breadth-first search) work?",
        "What are the ethical implications of using data analytics and algorithms to make decisions about individuals?",
        "How does facial recognition technology work?",
        "Describe the different types of software licenses (e.g., open source, proprietary).",
        "What are the key principles of Agile software development?",
        "How does a computer network route data packets between different devices?",
        "What are the challenges in managing and analyzing large datasets (Big Data)?",
        "Explain the different types of software testing (e.g., unit testing, integration testing, system testing).",
        "How does a distributed system (e.g., a cloud computing platform) operate?",
        "What are the major challenges in developing artificial general intelligence (AGI)?",
        "Describe the different approaches to data visualization.",
        "What are the key differences between relational and non-relational databases?",
        "How does a machine learning model learn from data?",
        "What are the challenges in developing effective methods for data privacy and security?",
        "How can technology be used to improve access to education in underserved communities?",
        "What are the potential benefits and risks of using gene therapy to treat genetic diseases?",
        "How can we design cities to be more sustainable and resilient to climate change?",
        "What are the major challenges in developing effective strategies for poverty reduction?",
        "How can we promote greater intercultural understanding and cooperation in a globalized world?",
        "What are the ethical considerations in using artificial intelligence in healthcare?",
        "How can we improve the effectiveness of international aid and development programs?",
        "What are the major challenges in achieving gender equality around the world?",
        "How can we address the problem of misinformation and disinformation online?",
        "What are the potential benefits and risks of using biotechnology in agriculture?",
        "How can we create a more inclusive and equitable society for people with disabilities?",
        "What are the major challenges in managing and protecting biodiversity?",
        "How can we promote greater access to healthcare in developing countries?",
        "What are the ethical implications of using social media data for research purposes?",
        "How can we improve the quality and accessibility of mental health services?",
        "What are the major challenges in addressing the global refugee crisis?",
        "How can we design transportation systems that are more sustainable and efficient?",
        "What are the ethical considerations in using drones for surveillance and warfare?",
        "How can we promote greater civic engagement and participation in democracy?",
        "What are the major challenges in combating corruption and promoting good governance?",
        "How can we use technology to improve disaster preparedness and response?",
        "What are the different strategies for managing and mitigating the risks of natural disasters?",
        "How can we promote more sustainable consumption and production patterns?",
        "What are the major challenges in addressing the problem of food insecurity?",
        "How can we improve the effectiveness of criminal justice systems?",
        "What are the ethical implications of using predictive policing algorithms?",
        "How can we promote greater access to clean water and sanitation in developing countries?",
        "What are the major challenges in managing and protecting the world's oceans?",
        "How can we design educational systems that better prepare students for the challenges of the 21st century?",
        "What are the ethical considerations in using genetic testing for personalized medicine?",
        "How can we design buildings and urban spaces that are more accessible and inclusive for people of all abilities?",
        "What are the major challenges in promoting peace and resolving conflicts around the world?",
        "How can we use technology to improve the efficiency and effectiveness of government services?",
        "What are the ethical implications of using artificial intelligence in the legal system?",
        "How can we promote greater understanding and tolerance of different religions and beliefs?",
        "What are the major challenges in addressing the problem of human trafficking?",
        "How can we design economic systems that are more equitable and sustainable?",
        "What are the ethical considerations in using social robots to care for the elderly or children?",
        "How can we promote greater access to affordable housing?",
        "What are the major challenges in managing and regulating the global financial system?",
        "How can we use technology to improve the quality and safety of food production?",
        "What are the ethical implications of using brain-computer interfaces?",
        "How can we promote greater respect for human rights around the world?",
        "What are the major challenges in addressing the problem of climate change denial?",
        "How can we design workplaces that are more inclusive and supportive of diversity?",
        "What are the ethical considerations in using autonomous weapons systems?",
        "How can we promote greater access to renewable energy sources?",
        "What are the major challenges in managing and preserving cultural heritage sites?",
        "How can we use technology to improve the lives of people living in poverty?",
        "What are the ethical implications of using data from wearable devices for health monitoring?",
        "How can we promote greater gender equality in the workplace?",
        "What are the major challenges in addressing the problem of antibiotic resistance?",
        "How can we design social media platforms that are less addictive and more conducive to well-being?",
        "What are the ethical considerations in using artificial intelligence for recruitment and hiring?",
        "How can we promote greater financial literacy and inclusion?",
        "What are the major challenges in managing and responding to pandemics?",
        "How can we use technology to improve the accuracy and fairness of elections?",
        "What are the ethical implications of using gene editing to enhance human capabilities?",
        "How can we promote greater animal welfare and protect endangered species?",
        "What are the major challenges in addressing the problem of plastic pollution?",
        "How can we design cities that are more resilient to natural disasters and extreme weather events?",
        "What are the ethical considerations in using artificial intelligence for personalized education?",
        "How can we promote greater access to justice for marginalized communities?",
        "What are the major challenges in managing and regulating the use of artificial intelligence?",
        "How can we use technology to improve the efficiency and sustainability of agriculture?",
            "What are some strategies to mitigate confirmation bias in decision-making, both individually and organizationally?",
        "How does cognitive dissonance influence our beliefs and behaviors, and how can we manage it?",
        "Explain the concept of emotional intelligence and its importance in personal and professional life.",
        "What are the different attachment styles, and how do they impact relationships?",
        "How can mindfulness and meditation practices improve mental well-being?",
        "Describe the different stages of grief and the coping mechanisms associated with each.",
        "What are the psychological effects of trauma, and what are some effective treatment approaches?",
        "Explain the different types of personality disorders and their characteristics.",
        "How do social norms and conformity influence individual behavior?",
        "What are the cognitive biases that can affect our judgment and decision-making?",
        "Describe the different types of learning (e.g., classical conditioning, operant conditioning, observational learning).",
        "What are the psychological factors that contribute to addiction, and what are some effective treatment strategies?",
        "Explain the concept of self-esteem and its impact on mental health.",
        "How does stress affect the body and mind, and what are some effective stress management techniques?",
        "What are the different types of memory (e.g., sensory, short-term, long-term), and how do they work?",
        "Describe the different stages of sleep and their functions.",
        "What are the psychological factors that influence motivation and goal-setting?",
        "Explain the different types of mental disorders (e.g., anxiety disorders, mood disorders, psychotic disorders).",
        "How does early childhood development impact later life outcomes?",
        "What are the ethical considerations in conducting psychological research?",
        "How do different leadership styles affect team performance and morale?",
        "What are the key factors that contribute to effective teamwork and collaboration?",
        "Describe the different stages of group development and the challenges associated with each.",
        "How can organizations foster a culture of innovation and creativity?",
        "What are the different types of organizational structures (e.g., functional, divisional, matrix) and their advantages and disadvantages?",
        "How can companies effectively manage organizational change and resistance to change?",
        "What are the ethical considerations in human resource management?",
        "Describe the different approaches to performance management and employee evaluation.",
        "How can organizations promote employee engagement and motivation?",
        "What are the major challenges in managing a diverse workforce?",
        "Explain the different theories of motivation (e.g., Maslow's hierarchy of needs, Herzberg's two-factor theory).",
        "How can organizations create a positive and supportive work environment?",
        "What are the different types of conflict that can arise in the workplace, and how can they be effectively managed?",
        "Describe the different approaches to employee training and development.",
        "How can organizations measure and improve employee satisfaction and well-being?",
        "What are the legal and ethical considerations in employee recruitment and selection?",
        "Explain the concept of organizational culture and its impact on employee behavior.",
        "How can organizations foster effective communication and collaboration across different departments and teams?",
        "What are the major challenges in managing remote teams and virtual work environments?",
        "How can organizations develop and implement effective leadership development programs?",
        "Describe the key elements of a strong brand identity.",
        "What are the different marketing channels (e.g., social media, email, search engine optimization) and their effectiveness?",
        "How can businesses effectively segment and target their customer base?",
        "What are the key principles of effective advertising and copywriting?",
        "Explain the different pricing strategies (e.g., cost-plus, value-based, competitive) and their applications.",
        "How can businesses measure the effectiveness of their marketing campaigns?",
        "What are the ethical considerations in marketing and advertising?",
        "Describe the different stages of the customer journey and how businesses can optimize each stage.",
        "How can businesses build strong customer relationships and loyalty?",
        "What are the major challenges in marketing in a digital age?",
        "Explain the concept of market research and its different methods (e.g., surveys, focus groups, experiments).",
        "How can businesses effectively manage their online reputation and brand image?",
        "What are the different types of sales strategies and techniques?",
        "Describe the key elements of a successful product launch.",
        "How can businesses use social media to build brand awareness and engage with customers?",
        "What are the legal and ethical considerations in online advertising?",
        "Explain the concept of search engine optimization (SEO) and how it works.",
        "How can businesses effectively use content marketing to attract and engage customers?",
        "What are the major challenges in international marketing and globalization?",
        "How can businesses use data analytics to improve their marketing and sales performance?"
    ]
    return random.sample(prompts, min(num_prompts, len(prompts)))

def sample_candidate_sentence(candidate_sentences, temperature=1.0):
    if not candidate_sentences:
        return ("", [])
    lengths = torch.tensor([length for (_, length) in candidate_sentences], dtype=torch.float)
    logits = -lengths / temperature
    probs = torch.nn.functional.softmax(logits, dim=0)
    sampled_index = torch.multinomial(probs, num_samples=1).item()
    return candidate_sentences[sampled_index]

def pick_sentence(sentences_lengths, soft):
    if not sentences_lengths:
        return ""
    if soft == 0:
        return min(sentences_lengths, key=lambda x: x[1])[0]
    return sample_candidate_sentence(sentences_lengths, temperature=soft)[0]

def build_multi_turn_messages(system_content, user_question, assistant_response=None):
    conversation = [{"role": "system", "content": system_content}]
    
    if assistant_response is None:
        conversation.append({"role": "user", "content": user_question})
    else:
        conversation.append({"role": "user", "content": user_question})
        conversation.append({"role": "assistant", "content": assistant_response})
        conversation.append({"role": "user", "content": "Please continue elaboration without repeating."})
    
    return conversation

# def batch_generate_next_sentence(
#     current_texts,
#     soft,
#     n_candidates=5,
#     max_new_tokens=50,
#     all_generated_sentences=None
# ):
#     if all_generated_sentences is None:
#         all_generated_sentences = [[] for _ in range(len(current_texts))]
#     data = []
#     for i, text in enumerate(current_texts):
#         system_content = (
#             "You are a helpful assistant that answers questions in multiple turns. "
#             "The first sentence should directly address the user's question; subsequent sentences elaborate "
#             "without repetition or restating the same sentence. "
#             "You must generate a single complete sentence."
#         )
        
#         conversation = build_multi_turn_messages(
#             system_content,
#             text,
#             None if not all_generated_sentences[i] else " ".join(all_generated_sentences[i])
#         )
        
#         data.append({'row_id': i, 'prompt': conversation})
#     prompts_df = pd.DataFrame(data)
#     config = {
#         "model_id": "gpt-4o-mini",
#         "num_completions": n_candidates,
#         "output": "batch_outputs.csv",
#         "temperature": 0.7,
#         "max_new_tokens": max_new_tokens
#     }
#     result_df = annotate_prompts(prompts_df, config)
#     chosen_sentences = []
#     for i in range(len(current_texts)):
#         row_data = result_df[result_df["row_id"] == i]
#         if row_data.empty:
#             chosen_sentences.append("")
#             continue
#         candidate_sentences = []
#         for c in range(1, n_candidates + 1):
#             col_name = f"response_{c}"
#             if col_name in row_data.columns:
#                 sentence = row_data[col_name].iloc[0] or ""
#                 idx = sentence.find('.')
#                 if idx != -1:
#                     sentence = sentence[:idx+1]
#                 candidate_sentences.append((sentence, len(sentence)))
#         chosen = pick_sentence(candidate_sentences, soft)
#         chosen_sentences.append(chosen)
#     return chosen_sentences

def batch_generate_next_sentence(
    current_texts,
    soft,
    n_candidates=5,
    max_new_tokens=50,
    all_generated_sentences=None
):
    if all_generated_sentences is None:
        all_generated_sentences = [[] for _ in range(len(current_texts))]


    sampling_params = SamplingParams(
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        max_tokens=max_new_tokens,
        repetition_penalty=1.05
    )

    prompts = []
    for i, text in enumerate(current_texts):
        system_content = (
            "You are a helpful assistant that answers questions in multiple turns. "
            "The first sentence should directly address the user's question; subsequent sentences elaborate "
            "without repetition or restating the same sentence. "
            "You must generate a single complete sentence."
        )
        
        conversation = build_multi_turn_messages(
            system_content,
            text,
            None if not all_generated_sentences[i] else " ".join(all_generated_sentences[i])
        )
        
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
        prompts.append(prompt)

    outputs = model.generate(prompts, sampling_params)
    
    chosen_sentences = []
    for i, output in enumerate(outputs):
        candidate_sentences = []
        for text in output.outputs[:n_candidates]:
            sentence = text.text.strip()
            idx = sentence.find('.')
            if idx != -1:
                sentence = sentence[:idx+1]
            candidate_sentences.append((sentence, len(sentence)))
        
        chosen = pick_sentence(candidate_sentences, soft)
        chosen_sentences.append(chosen)
    
    return chosen_sentences

def batch_shortest_path_generation(prompts, n_candidates=5, max_sentences=5, max_tokens_per_sentence=50, soft=0.7):
    current_texts = prompts[:]
    all_generated_sentences = [[] for _ in range(len(prompts))]
    for _ in range(max_sentences):
        next_sents = batch_generate_next_sentence(
            current_texts, soft, n_candidates, max_tokens_per_sentence, all_generated_sentences
        )
        got_new_sentence = False
        for i, sent in enumerate(next_sents):
            if sent.strip():
                got_new_sentence = True
                all_generated_sentences[i].append(sent)
                current_texts[i] += " " + sent
        if not got_new_sentence:
            break
    return current_texts, all_generated_sentences

def single_candidate_shortest_path(prompts, max_sentences=5, max_tokens_per_sentence=50):
    current_texts = prompts[:]
    all_sents = [[] for _ in range(len(prompts))]
    for _ in range(max_sentences):
        next_sents = batch_generate_next_sentence(
            current_texts, soft=0, n_candidates=1, max_new_tokens=max_tokens_per_sentence, all_generated_sentences=all_sents
        )
        got_new_sentence = False
        for i, sent in enumerate(next_sents):
            if sent.strip():
                got_new_sentence = True
                all_sents[i].append(sent)
                current_texts[i] += " " + sent
        if not got_new_sentence:
            break
    return current_texts, all_sents

# def main():
#     max_tokens_per_sentence = 20
#     max_sentences = 5
#     n_candidates = 10
#     raw_prompts = load_prompts(num_prompts=100)
#     final_texts_sbon, all_sentences_sbon = batch_shortest_path_generation(
#         raw_prompts, n_candidates, max_sentences, max_tokens_per_sentence, soft=1.2
#     )
#     final_texts_bon, all_sentences_bon = batch_shortest_path_generation(
#         raw_prompts, n_candidates, max_sentences, max_tokens_per_sentence, soft=0
#     )
#     final_texts_single, all_sentences_single = single_candidate_shortest_path(
#         raw_prompts, max_sentences, max_tokens_per_sentence
#     )
#     with open("outputs.csv", "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["prompt", "sBonResponse", "BoNResponse", "GreedyResponse"])
#         for i, prompt_text in enumerate(raw_prompts):
#             writer.writerow([
#                 prompt_text,
#                 final_texts_sbon[i][final_texts_sbon[i].find('?') + 1:],
#                 final_texts_bon[i][final_texts_bon[i].find('?') + 1:],
#                 final_texts_single[i][final_texts_single[i].find('?') + 1:],
#             ])

# if __name__ == "__main__":
#     main()

def main():
    max_tokens_per_sentence = 20
    max_sentences = 5
    raw_prompts = load_prompts(num_prompts=200)
    model_name = "meta-llama/Llama-3.2-1B"
    
    n_values = [2, 4, 8, 12, 18, 24, 32]
    soft_values = [0.1, 0.35, 0.6, 0.85, 1.1, 1.35, 1.6, 1.85, 2.1]
    
    for n in n_values:
        for soft in soft_values:
            final_texts_sbon, all_sentences_sbon = batch_shortest_path_generation(
                raw_prompts, n, max_sentences, max_tokens_per_sentence, soft
            )
            final_texts_bon, all_sentences_bon = batch_shortest_path_generation(
                raw_prompts, n, max_sentences, max_tokens_per_sentence, soft=0
            )
            final_texts_single, all_sentences_single = single_candidate_shortest_path(
                raw_prompts, max_sentences, max_tokens_per_sentence
            )
            
            output_filename = f"output_{model_name.replace('/', '_')}_{n}_{soft}.csv"
            with open(output_filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["prompt", "sBonResponse", "BoNResponse", "GreedyResponse"])
                for i, prompt_text in enumerate(raw_prompts):
                    writer.writerow([
                        prompt_text,
                        final_texts_sbon[i][final_texts_sbon[i].find('?') + 1:],
                        final_texts_bon[i][final_texts_bon[i].find('?') + 1:],
                        final_texts_single[i][final_texts_single[i].find('?') + 1:],
                    ])
            print(f"Saved: {output_filename}")

if __name__ == "__main__":
    main()

