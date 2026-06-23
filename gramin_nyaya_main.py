from stt_service import record_and_transcribe
from rag_logic import ask_gramin_nyaya
import sys

def start_system():
    print("\n" + "="*45)
    print("GRAMIN-NYAYA: AI LEGAL ASSISTANT (OFFLINE)")
    print("="*45)
    
    print("\nपद्धति चुनें (Choose Input Mode):")
    print("1. टाइप करें (Type question)")
    print("2. बोलें (Voice input)")
    mode = input("अपना विकल्प दर्ज करें [1 या 2] (Default is 1): ").strip()
    if mode not in ["1", "2"]:
        mode = "1"
        
    while True:
        try:
            choice = input("\nप्रेस करें 'Enter' सवाल पूछने के लिए (या 'exit' लिखें): ")
            if choice.lower() == 'exit':
                print("\nधन्यवाद! ग्रामिन न्याय प्रणाली बंद हो रही है। (Thank you! Shutting down.)")
                break
            
            if mode == "2":
                user_text = record_and_transcribe()
            else:
                user_text = input("सवाल टाइप करें (Type question): ")
            
            print(f"\nआपका सवाल (Your Question): {user_text}")
            
            if not user_text or len(user_text.strip()) < 3:
                print("कृपया स्पष्ट सवाल पूछें। (Please ask a clear question.)")
                continue
            
            print("\nसोच रहा हूँ... (Thinking...)")
            
            # Added error handling to prevent crashes during testing
            try:
                answer = ask_gramin_nyaya(user_text)
                
                print("\n" + "-"*45)
                print("कानूनी परामर्श (Legal Advice):")
                print(answer)
                print("-" * 45)
                
            except Exception as e:
                print("\n[त्रुटि/Error] जवाब लाने में समस्या हुई:")
                print(str(e))
                print("कृपया दोबारा प्रयास करें। (Please try again.)")
                
        except KeyboardInterrupt:
            # Handles Ctrl+C gracefully without throwing a massive traceback
            print("\n\nप्रणाली को जबरन बंद किया गया। (System forcibly closed.) अलविदा!")
            sys.exit(0)

if __name__ == "__main__":
    start_system()