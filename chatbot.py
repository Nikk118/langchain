from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage


load_dotenv()

def main():
    print("Chatbot Staring up,please wait...\n")
    llm=HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        task="text-generation",
        max_new_tokens=200,
        temperature=0.7,
    )

    chat_model=ChatHuggingFace(llm=llm)
    history = [
    SystemMessage(content="""You are a helpful assistant. 
Rules you must follow:
- For simple/casual questions (greetings, basic definitions): reply in 1-2 sentences max
- For medium questions: reply in 3-5 sentences max  
- For complex technical questions: reply in as much detail as needed
- Never list every possible meaning of a word unless asked
- Never add summaries or sign-offs at the end
- Just answer directly, nothing else""")
]

    print("ready|Type 'quit to exit ..\n")

    while True:
        try:
            userInput=input("you: ").strip()
            if not userInput:
                continue
            if userInput.lower() in ("quit","exit","bye"):
                break
            history.append(HumanMessage(content=userInput))
            print("Bot: ",end="",flush=True)
            full_reply=""
            for chunk in chat_model.stream(history):
                text=chunk.content
                print(text,end="",flush=True)
                full_reply+=text 
            print("\n")
            history.append(AIMessage(content=full_reply))
            if len(history)>10:
                history=[history[0]]+history[-5:]


        except KeyboardInterrupt:
            print("\nBot: Goodbye!")
            break

        except Exception as e:
            print(f"\n Error: {e}")
            print("Something went wrong. Try again or check your token.\n")
if __name__=="__main__":
    main()