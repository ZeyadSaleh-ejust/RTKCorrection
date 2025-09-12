from fastapi import FastAPI
from routes import base,data
import asyncpg
from helpers.config import get_settings



app = FastAPI()

@app.on_event("startup")
async def startup_span():
    setting = get_settings()

    # Establishing connection
    connection_string = f"{setting.POSTGRESQL_URL}/{setting.POSTGRESQL_DATABASE}"
    app.postgre_sql_pool = await asyncpg.create_pool(connection_string,
                                                    min_size=5,
                                                    max_size=20)

@app.on_event("shutdown")
async def shutdown_span():
    await app.postgre_sql_pool.close()


app.include_router(base.base_router)
app.include_router(data.data_router)

