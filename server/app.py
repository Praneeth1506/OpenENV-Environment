# server/app.py
# Step 4 of OpenEnv structure — FastAPI server

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from server.environment import SafeSignalEnvironment

# Create environment instance
env = SafeSignalEnvironment()

# Try to create the app — we will update this once we know
# the exact openenv function name
try:
    import openenv
    # Try common patterns
    if hasattr(openenv.openenv, 'create_fastapi_app'):
        app = openenv.openenv.create_fastapi_app(env)
    elif hasattr(openenv, 'create_fastapi_app'):
        app = openenv.create_fastapi_app(env)
    else:
        print("Available in openenv.openenv:", dir(openenv.openenv))
        print("Find the server creation function above")
        app = None

except Exception as e:
    print(f"Error creating app: {e}")
    app = None

if __name__ == "__main__":
    if app:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("App not created — check openenv function name")