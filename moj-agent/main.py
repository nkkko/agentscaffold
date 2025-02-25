"""Main entry point for moj-agent agent."""

import os
import json
import asyncio
from moj_agent import Agent
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


async def main():
    """Run the agent."""
    agent = Agent()
    
    # Set your Daytona environment variables in .env file
    # Alternatively, you can set them here:
    # os.environ["DAYTONA_API_KEY"] = "your-api-key"
    # os.environ["DAYTONA_SERVER_URL"] = "your-server-url"
    
    # Demo message
    message = "Hello, I'd like to know more about moj-agent."
    
    print(f"Sending message: {message}")
    print("Processing with MojAgent agent...")
    
    result = await agent.run({"message": message})
    
    print("\nResponse:")
    print(f"  {result['response']}")
    
    if result['metadata']:
        print("\nMetadata:")
        for key, value in result['metadata'].items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())