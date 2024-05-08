import threading
import subprocess
import sys

def run_script(script_name):
    subprocess.run([sys.executable, script_name])

# Names of your python scripts
# script1 = 'voice_test.py'
# script2 = 'face_emotion.py'

script1 = 'voiceemoloop.py'
script2 = 'faceemoloop.py'
script3 = 'recommender.py'

# Create two threads
thread1 = threading.Thread(target=run_script, args=(script1,))
thread2 = threading.Thread(target=run_script, args=(script2,))

# Start the threads
thread1.start()
thread2.start()

# Wait for both threads to finish
thread1.join()
thread2.join()

print("Both scripts have finished executing.")

thread3 = threading.Thread(target=run_script, args=(script3,))
thread3.start()
thread3.join()





