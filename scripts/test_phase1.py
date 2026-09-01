import sys
import os
from datetime import datetime

# Include project root in python search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    print("="*60)
    print("AI-QTriage - Running Phase 1 Backend Verification")
    print("="*60)
    
    try:
        from backend.database.connection import get_database, init_db_indexes
        
        print("[1/3] Testing MongoDB Connection...")
        db = get_database()
        print(f"      Success! Connected to database: '{db.name}'")
        
        print("[2/3] Checking collection index creation...")
        init_db_indexes()
        print("      Success! Indexes built.")
        
        print("[3/3] Verifying collection read/write/delete operations...")
        test_case_id = "test-phase1-validation"
        test_case = {
            "case_id": test_case_id,
            "created_at": datetime.utcnow(),
            "status": "temporary_test_case"
        }
        
        # Clean up existing test runs if any
        db.cases.delete_many({"case_id": test_case_id})
        
        # Write
        db.cases.insert_one(test_case)
        print("      - Inserted test case document.")
        
        # Read
        retrieved = db.cases.find_one({"case_id": test_case_id})
        assert retrieved is not None, "Failed to retrieve the inserted test case."
        assert retrieved["status"] == "temporary_test_case", "Document content mismatch."
        print("      - Verified retrieval of written case.")
        
        # Delete
        db.cases.delete_one({"case_id": test_case_id})
        print("      - Cleaned up test case document.")
        
        print("\nAll database checks passed successfully!")
        print("="*60)
        sys.exit(0)
        
    except Exception as e:
        print(f"\nERROR: Verification failed: {str(e)}", file=sys.stderr)
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()
