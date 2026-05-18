from stt_service import record_and_transcribe
from rag_logic import ask_gramin_nyaya
import sys

def start_system():
    print("\n" + "="*45)
    print("GRAMIN-NYAYA: AI LEGAL ASSISTANT (OFFLINE)")
    print("="*45)
    
    while True:
        choice = input("\nप्रेस करें 'Enter' सवाल पूछने के लिए (या 'exit' लिखें): ")
        if choice.lower() == 'exit':
            break
        
        # NOTE: Using text input for testing accuracy. 
        # To use voice, uncomment the line below and comment the input() line.
        # user_text = record_and_transcribe() 
        user_text = input("सवाल टाइप करें: ")
        
        print(f"\nआपका सवाल: {user_text}")
        
        if len(user_text.strip()) < 3:
            print("कृपया स्पष्ट सवाल पूछें।")
            continue
            
        answer = ask_gramin_nyaya(user_text)
        
        print("\n" + "-"*30)
        print("कानूनी परामर्श (Legal Advice):")
        print(answer)
        print("-"*30)

if __name__ == "__main__":
    start_system()