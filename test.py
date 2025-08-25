import pyttsx3

engine = pyttsx3.init(driverName="sapi5")   # Windows: sapi5, Linux: espeak, Mac: nsss
voices = engine.getProperty("voices")

print("Available voices:")
for i, v in enumerate(voices):
    print(i, v.name)

engine.setProperty("voice", voices[0].id)  # pick first voice
engine.setProperty("rate", 150)

print("🔊 Speaking now...")
engine.say("Hello Tarun, this is a test of the SAMVAAD project.")
engine.runAndWait()
