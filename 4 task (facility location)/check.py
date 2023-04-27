import asyncio

async def long_running_function():
    for i in range(10):
        print(f"Long running function - iteration {i}")
        await asyncio.sleep(1)

async def monitor_function():
    for i in range(5):
        print(f"Monitoring function - iteration {i}")
        await asyncio.sleep(2)
    
    # stop the long running function after 5 iterations
    #long_running_function_task.cancel()

# start the long running function and monitor it asynchronously
loop = asyncio.get_event_loop()
long_running_function_task = loop.create_task(long_running_function())
monitor_task = loop.create_task(monitor_function())

# run the loop until both tasks complete
loop.run_until_complete(asyncio.gather(long_running_function_task, monitor_task))