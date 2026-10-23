from inference.predict_review import predict_review

review = """Pleasant staff and housekeeping. Above average breakfast. The rooms were clean and the staff made sure we had fresh linnens. It was bedbug free!! I would definitely stay there again"""

result = predict_review(review)

print("\nProblematic Review Test")
print("=" * 50)

for key, value in result.items():
    print(f"{key}: {value}")