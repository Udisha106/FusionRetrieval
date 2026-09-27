from src.query.query_pipeline import QueryPipeline


pipeline = QueryPipeline()

query = "find code that calculates the mean of numbers"

result = pipeline.process(query)

print("\n===== QUERY PIPELINE =====")

print("\nOriginal:")
print(result.original)

print("\nExpanded:")
print(result.expanded)

print("\nHyDE:")
print(result.hyde)

print("\nIntent:")
print(result.intent)

print("\nEntities:")
print(result.entities)

print("\nIdentifiers:")
print(result.identifiers)

print("\n===== VIEWS FOR RETRIEVAL =====")

for view in result.views_for_retrieval():
    print("\n", view)