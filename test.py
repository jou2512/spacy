import re
import codecs
import spacy
import os
import openai

from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Access the API key from the environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# initate messages list with system instructions
messages_origin = [
    {"role": "system", "content": "You are a linguistic Expert."},
    {"role": "user", "content": "Process large text in to different topic paragraphs by comparing the given sentence to the previous one, if it's about the same detailed topic it's still the same paragraph, if not it's a new one. Make sure to detect every nuanced changing of topic and be very detailed in checking. Sentence by sentence, you will exactly and only output 'true' or 'false'. If it's still the same paragraph you output 'true' if not you will output 'false'. The first sentence given, is obviously the start of the first paragraph, so always return true, then sentence by sentence compare it to the current paragraph. The current paragraph are all the sentence which output was 'true' to be the same paragraph. If a 'false' accrues, this is the start of the new paragraph. Only return 'true' or 'false'. Stop doing that when the end signal '###STOP###' gets inputted, then return a list of the conversation. Answer with 'okay' and wait for the first sentence."},
    {"role": "assistant", "content": "okay"},
]

messages = messages_origin.copy()

# Add a message to the conversation history
def add_message(role, content, messages):
    messages.append({"role": role, "content": content})
    return messages

# Get the response from the assistant
def get_response(prompt, messages):

    # Add the prompt to the messages
    messages = add_message("user", prompt, messages)

    # Get the response from the assistant
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=messages,
    )

    # Add the response to the messages
    messages = add_message(
        "assistant", response.choices[0].message.content, messages)
    return messages

# Process a list of sentences
def process_sentences(sentences_new, messages):
    paragraphs = []
    sentences_copy = sentences_new.copy()
    for i, sentence in enumerate(sentences_copy):
        messages = get_response(sentence, messages)
        if "false" in messages[-1]['content'] and i>0:
            if ''.join(sentences_copy[:i-1]) != '': 
                paragraphs.append(''.join(sentences_copy[:i-1]))
            sentences_copy = sentences_copy[i-1:]
            messages = messages_origin.copy()
    if ''.join(sentences_copy) != '':
        paragraphs.append(''.join(sentences_copy))
    return paragraphs


def process_text(text, messages, i):
    nlp = spacy.load("de_core_news_sm")
    doc = nlp(text)
    assert doc.has_annotation("SENT_START")
    listsentences = [sent.text for sent in doc.sents]
    paragraphs = process_sentences(listsentences, messages)

    if i == 0:
        return paragraphs

    result = []
    for j, p in enumerate(paragraphs):
        messages = messages_origin.copy()
        subresult = process_text(p, messages, i-1)
        result.append(subresult)

    return result


def print_nested_strings(nested_list):
    for item in nested_list:
        if isinstance(item, str):
            print(item,"\n\n")
        else:
            print_nested_strings(item)


nlp = spacy.load("de_core_news_sm")
nlp.enable_pipe("senter")
# Merge noun phrases and entities for easier analysis
nlp.add_pipe("merge_entities")
nlp.add_pipe("merge_noun_chunks")

# read in the school paper text file but encoding it with utf-8
with codecs.open("school_paper.txt", "r", "utf-8") as f:
    text = f.read()
# remove newline characters and make all text lowercase
text = re.sub(r"\n", "", text)
text = re.sub(r"\r", "", text)
doc = nlp(text)
#print([(e.text, e.label_) for e in doc.ents])


assert doc.has_annotation("SENT_START")
listsentences = [sent.text for sent in doc.sents]

#paragraphs = process_sentences(listsentences, messages)

result = process_text(text, messages, 1)
#print(result)

print_nested_strings(result)
# remove the first two items of the messages list
#messages = messages[2:]

# Create tuples of user and assistant messages
#user_messages = []
#assistant_messages = []
#for i, message in enumerate(messages):
#    if message['role'] == 'user':
#        user_messages.append(message['content'])
#        if i+1 < len(messages) and messages[i+1]['role'] == 'assistant':
#            assistant_messages.append(messages[i+1]['content'])
#
# Print the tuples
#for user_msg, assistant_msg in zip(user_messages, assistant_messages):
#    print('User:', user_msg)
#    print('Assistant:', assistant_msg)

#paragraphs = []
#current_paragraph = ""
#for user_msg, assistant_msg in zip(user_messages, assistant_messages):
#    if "true" in assistant_msg:
#        current_paragraph += user_msg + " "
#    else:
#        paragraphs.append(current_paragraph.strip())
#        current_paragraph = user_msg
#if current_paragraph != "":
#    paragraphs.append(current_paragraph.strip())
#
#for paragraph in paragraphs:
#    print("")
#    print(paragraph)
#    print("")
#for i, sent in enumerate(doc.sents):
#    print(sent.text, "\n\n")

#tok_exp = nlp.tokenizer.explain(text)
#assert [t.text for t in doc if not t.is_space] == [t[1] for t in tok_exp]
#for t in tok_exp:
#    print(t[1], "\t", t[0])
#print([w.text for w in doc]
