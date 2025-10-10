"""PrometheusULTIMATE CLI."""

import click
import json
import os
import sys
from pathlib import Path
from typing import Optional
import httpx
import asyncio
from datetime import datetime


class PromuCLI:
    """Main CLI class."""
    
    def __init__(self, gateway_url: str = "http://localhost:8090"):
        self.gateway_url = gateway_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
    
    async def health_check(self) -> bool:
        """Check if services are healthy."""
        try:
            response = await self.client.get(f"{self.gateway_url}/health")
            return response.status_code == 200
        except Exception:
            return False
    
    async def create_task(self, description: str, project: str = "default", 
                         time_limit: int = 300, cost_limit: float = 1.0) -> dict:
        """Create a new task."""
        task_data = {
            "description": description,
            "project": project,
            "time_limit": time_limit,
            "cost_limit": cost_limit
        }
        
        response = await self.client.post(f"{self.gateway_url}/task", json=task_data)
        response.raise_for_status()
        return response.json()
    
    async def get_task(self, task_id: str) -> dict:
        """Get task details."""
        response = await self.client.get(f"{self.gateway_url}/task/{task_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_task_logs(self, task_id: str) -> list:
        """Get task logs."""
        response = await self.client.get(f"{self.gateway_url}/task/{task_id}/logs")
        response.raise_for_status()
        return response.json()
    
    async def save_memory(self, content: str, project: str = "default", 
                         content_type: str = "text") -> dict:
        """Save content to memory."""
        memory_data = {
            "content": content,
            "project": project,
            "content_type": content_type
        }
        
        response = await self.client.post(f"{self.gateway_url}/memory/save", json=memory_data)
        response.raise_for_status()
        return response.json()
    
    async def search_memory(self, query: str, project: str = "default", 
                           k: int = 5) -> list:
        """Search memory."""
        params = {
            "query": query,
            "project": project,
            "k": k
        }
        
        response = await self.client.get(f"{self.gateway_url}/memory/search", params=params)
        response.raise_for_status()
        return response.json()
    
    async def register_skill(self, skill_path: str, dry_run: bool = False) -> dict:
        """Register a skill."""
        if not os.path.exists(skill_path):
            raise FileNotFoundError(f"Skill path not found: {skill_path}")
        
        skill_data = {
            "skill_path": skill_path,
            "dry_run": dry_run
        }
        
        response = await self.client.post(f"{self.gateway_url}/skill/register", json=skill_data)
        response.raise_for_status()
        return response.json()
    
    async def list_skills(self) -> list:
        """List registered skills."""
        response = await self.client.get(f"{self.gateway_url}/skill/list")
        response.raise_for_status()
        return response.json()
    
    async def run_skill(self, skill_name: str, inputs: dict) -> dict:
        """Run a skill."""
        skill_data = {
            "skill_name": skill_name,
            "inputs": inputs
        }
        
        response = await self.client.post(f"{self.gateway_url}/skill/run", json=skill_data)
        response.raise_for_status()
        return response.json()


# CLI instance
cli = PromuCLI()


@click.group()
@click.option('--gateway-url', default='http://localhost:8090', 
              help='Gateway URL')
@click.pass_context
def main(ctx, gateway_url):
    """PrometheusULTIMATE CLI - AGI Engineering Platform."""
    ctx.ensure_object(dict)
    ctx.obj['gateway_url'] = gateway_url
    cli.gateway_url = gateway_url


@main.command()
@click.argument('description')
@click.option('--project', default='default', help='Project name')
@click.option('--time', default=300, help='Time limit in seconds')
@click.option('--cost', default=1.0, help='Cost limit in USD')
@click.pass_context
def task(ctx, description, project, time, cost):
    """Create a new task."""
    async def _create_task():
        try:
            if not await cli.health_check():
                click.echo("❌ Services are not healthy. Please check if services are running.", err=True)
                return
            
            click.echo(f"🚀 Creating task: {description}")
            result = await cli.create_task(description, project, time, cost)
            
            click.echo(f"✅ Task created successfully!")
            click.echo(f"   Task ID: {result['task_id']}")
            click.echo(f"   Status: {result['status']}")
            click.echo(f"   Project: {result['project']}")
            
        except Exception as e:
            click.echo(f"❌ Error creating task: {e}", err=True)
        finally:
            await cli.close()
    
    asyncio.run(_create_task())


@main.command()
@click.argument('task_id')
@click.pass_context
def logs(ctx, task_id):
    """Get task logs."""
    async def _get_logs():
        try:
            if not await cli.health_check():
                click.echo("❌ Services are not healthy.", err=True)
                return
            
            click.echo(f"📋 Getting logs for task: {task_id}")
            logs = await cli.get_task_logs(task_id)
            
            if not logs:
                click.echo("No logs found for this task.")
                return
            
            for log in logs:
                timestamp = log.get('timestamp', 'Unknown')
                level = log.get('level', 'INFO')
                message = log.get('message', '')
                click.echo(f"[{timestamp}] {level}: {message}")
                
        except Exception as e:
            click.echo(f"❌ Error getting logs: {e}", err=True)
        finally:
            await cli.close()
    
    asyncio.run(_get_logs())


@main.command()
@click.argument('content')
@click.option('--project', default='default', help='Project name')
@click.option('--type', 'content_type', default='text', help='Content type')
@click.pass_context
def mem_save(ctx, content, project, content_type):
    """Save content to memory."""
    async def _save_memory():
        try:
            if not await cli.health_check():
                click.echo("❌ Services are not healthy.", err=True)
                return
            
            click.echo(f"💾 Saving to memory: {content[:50]}...")
            result = await cli.save_memory(content, project, content_type)
            
            click.echo(f"✅ Content saved to memory!")
            click.echo(f"   Memory ID: {result.get('memory_id', 'N/A')}")
            click.echo(f"   Project: {project}")
            
        except Exception as e:
            click.echo(f"❌ Error saving to memory: {e}", err=True)
        finally:
            await cli.close()
    
    asyncio.run(_save_memory())


@main.command()
@click.argument('query')
@click.option('--project', default='default', help='Project name')
@click.option('--k', default=5, help='Number of results')
@click.pass_context
def mem_search(ctx, query, project, k):
    """Search memory."""
    async def _search_memory():
        try:
            if not await cli.health_check():
                click.echo("❌ Services are not healthy.", err=True)
                return
            
            click.echo(f"🔍 Searching memory: {query}")
            results = await cli.search_memory(query, project, k)
            
            if not results:
                click.echo("No results found.")
                return
            
            click.echo(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                content = result.get('content', '')
                score = result.get('score', 0)
                click.echo(f"{i}. [Score: {score:.3f}] {content[:100]}...")
                
        except Exception as e:
            click.echo(f"❌ Error searching memory: {e}", err=True)
        finally:
            await cli.close()
    
    asyncio.run(_search_memory())


@main.command()
@click.argument('skill_path')
@click.option('--dry-run', is_flag=True, help='Validate without registering')
@click.pass_context
def skill_add(ctx, skill_path, dry_run):
    """Add a skill."""
    async def _add_skill():
        try:
            if not await cli.health_check():
                click.echo("❌ Services are not healthy.", err=True)
                return
            
            action = "Validating" if dry_run else "Registering"
            click.echo(f"🔧 {action} skill: {skill_path}")
            
            result = await cli.register_skill(skill_path, dry_run)
            
            if dry_run:
                if result.get('valid', False):
                    click.echo("✅ Skill validation passed!")
                    click.echo(f"   Name: {result.get('name', 'N/A')}")
                    click.echo(f"   Version: {result.get('version', 'N/A')}")
                else:
                    click.echo("❌ Skill validation failed!")
                    for error in result.get('errors', []):
                        click.echo(f"   Error: {error}")
            else:
                click.echo("✅ Skill registered successfully!")
                click.echo(f"   Name: {result.get('name', 'N/A')}")
                click.echo(f"   Version: {result.get('version', 'N/A')}")
                
        except Exception as e:
            click.echo(f"❌ Error with skill: {e}", err=True)
        finally:
            await cli.close()
    
    asyncio.run(_add_skill())


@main.command()
@click.pass_context
def skill_list(ctx):
    """List registered skills."""
    async def _list_skills():
        try:
            if not await cli.health_check():
                click.echo("❌ Services are not healthy.", err=True)
                return
            
            click.echo("📋 Registered skills:")
            skills = await cli.list_skills()
            
            if not skills:
                click.echo("No skills registered.")
                return
            
            for skill in skills:
                name = skill.get('name', 'Unknown')
                version = skill.get('version', 'Unknown')
                description = skill.get('description', 'No description')
                click.echo(f"  • {name} v{version}")
                click.echo(f"    {description}")
                click.echo()
                
        except Exception as e:
            click.echo(f"❌ Error listing skills: {e}", err=True)
        finally:
            await cli.close()
    
    asyncio.run(_list_skills())


@main.command()
@click.argument('skill_name')
@click.argument('inputs_json')
@click.pass_context
def skill_run(ctx, skill_name, inputs_json):
    """Run a skill with inputs."""
    async def _run_skill():
        try:
            if not await cli.health_check():
                click.echo("❌ Services are not healthy.", err=True)
                return
            
            # Parse inputs JSON
            try:
                inputs = json.loads(inputs_json)
            except json.JSONDecodeError as e:
                click.echo(f"❌ Invalid JSON inputs: {e}", err=True)
                return
            
            click.echo(f"🏃 Running skill: {skill_name}")
            result = await cli.run_skill(skill_name, inputs)
            
            if result.get('success', False):
                click.echo("✅ Skill executed successfully!")
                outputs = result.get('outputs', {})
                for key, value in outputs.items():
                    click.echo(f"   {key}: {value}")
            else:
                click.echo("❌ Skill execution failed!")
                click.echo(f"   Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            click.echo(f"❌ Error running skill: {e}", err=True)
        finally:
            await cli.close()
    
    asyncio.run(_run_skill())


@main.command()
@click.pass_context
def health(ctx):
    """Check service health."""
    async def _check_health():
        try:
            click.echo("🏥 Checking service health...")
            
            if await cli.health_check():
                click.echo("✅ Services are healthy!")
            else:
                click.echo("❌ Services are not responding.")
                
        except Exception as e:
            click.echo(f"❌ Health check failed: {e}", err=True)
        finally:
            await cli.close()
    
    asyncio.run(_check_health())


if __name__ == '__main__':
    main()
