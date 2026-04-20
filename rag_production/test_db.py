from rag_demo.core import ProductionDatabase, RAGConfig


def main() -> None:
    config = RAGConfig()
    config.validate()

    db = ProductionDatabase(config)

    try:
        result = db.health_check()

        print("DB Health Check")
        print("----------------")
        for key, value in result.items():
            print(f"{key}: {value}")

    finally:
        db.close()


if __name__ == "__main__":
    main()