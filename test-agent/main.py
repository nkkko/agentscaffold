#!/usr/bin/env python
"""Main entry point for test-agent with mandatory Daytona execution."""

import os
import sys
import json
import asyncio
import argparse
from typing import Dict, Any, Optional

# Environment setup
try:
    from dotenv import load_dotenv
    # Try different locations for .env file
    env_paths = ['.env', '../.env']
    env_loaded = False
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            env_loaded = True
            break
    
    if env_loaded:
        print("✅ Loaded environment variables")
    else:
        print("⚠️ No .env file found")
except ImportError:
    print("⚠️ python-dotenv not installed")

# Verify Daytona config
daytona_api_key = os.environ.get("DAYTONA_API_KEY")
if not daytona_api_key:
    print("❌ DAYTONA_API_KEY not set!")
else:
    key_preview = f"***{daytona_api_key[-4:]}" if len(daytona_api_key) > 4 else "****"
    print(f"✅ Loaded .env (DAYTONA_API_KEY={key_preview})")
    
if not os.environ.get("DAYTONA_SERVER_URL"):
    print("❌ DAYTONA_SERVER_URL not set!")

try:
    from test_agent.agent import Agent
except ImportError as e:
    print(f"❌ Error importing agent: {e}")
    print("Make sure the package is installed (pip install -e .)")
    sys.exit(1)

async def process_single_message(agent, message: str) -> Dict[str, Any]:
    """Process a single message with the agent."""
    try:
        result = await agent.run({"message": message})
        return result
    except Exception as e:
        print(f"❌ Error processing message: {e}")
        return {"response": f"Error: {str(e)}", "metadata": {"error": True}}

async def run_interactive(agent):
    """Interactive session with Daytona."""
    print(f"\n🚀 {agent.name} - {agent.description}")
    print("🔒 Running in Daytona cloud environment")
    print("Type 'quit' or 'exit' to end session\n")
    
    while True:
        try:
            message = input("You: ").strip()
            if not message:
                continue
                
            if message.lower() in ["quit", "exit", "bye"]:
                print("\n👋 Ending session...")
                break
                
            print("⚡ Processing...")
            result = await process_single_message(agent, message)
            
            print(f"\nAgent: {result.get('response')}")
            
            # Display any search results if available
            if result.get("search_results"):
                print("\nSearch Results:")
                for i, result in enumerate(result.get("search_results"), 1):
                    print(f"{i}. {result.get('title')}: {result.get('url')}")
            
            # Display memory context if available
            if result.get("memory_context"):
                print("\nRetrieved from memory:")
                print(result.get("memory_context"))
                
        except KeyboardInterrupt:
            print("\n👋 Ending session...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

async def main(input_data: Optional[Dict[str, Any]] = None):
    """
    Main entry point with Daytona enforcement.
    
    Args:
        input_data: Optional input data for the agent
        
    Returns:
        Agent output
    """
    parser = argparse.ArgumentParser(description="Run the test-agent agent")
    parser.add_argument("--message", "-m", help="Message to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    
    parser.add_argument("--search", "-s", help="Search query to use")
    
    
    
    parser.add_argument("--context", "-c", action="store_true", help="Retrieve context from memory")
    parser.add_argument("--context-query", help="Query for context retrieval")
    parser.add_argument("--store", action="store_true", help="Store conversation in memory")
    
    
    args = parser.parse_args()
    
    try:
        print("🔧 Initializing Daytona agent...")
        agent = Agent()
        
        # If input_data is provided, use it directly
        if input_data:
            return await agent.run(input_data)
        
        # Create input data from arguments
        if args.message:
            input_data = {"message": args.message}
            
            # Add optional parameters if provided
            
            if args.search:
                input_data["search_query"] = args.search
            
            
            
            if args.context:
                input_data["retrieve_context"] = True
            if args.context_query:
                input_data["context_query"] = args.context_query
            if args.store:
                input_data["store_in_memory"] = True
            
            
            result = await agent.run(input_data)
            print(f"\nAgent: {result.get('response')}")
            return result
        else:
            # Run in interactive mode
            await run_interactive(agent)
            return {"response": "Interactive session completed", "metadata": {"interactive": True}}
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"response": f"Error: {str(e)}", "metadata": {"error": True}}

if __name__ == "__main__":
    result = asyncio.run(main())