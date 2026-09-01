import sys
import os
import socket
import ssl
import dns.resolver
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, OperationFailure
import urllib.parse

# Set search path to include project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import settings

def run_diagnostics():
    print("====================================================")
    print("MONGODB DIAGNOSTICS")
    print("====================================================")
    
    uri = settings.mongodb_uri
    db_name = settings.mongodb_database
    print(f"Configured Database: {db_name}")
    
    # Mask password for printing
    parsed_uri = urllib.parse.urlparse(uri)
    password_masked = uri
    if parsed_uri.password:
        password_masked = uri.replace(parsed_uri.password, "****")
    print(f"Configured URI: {password_masked}")
    
    # 1. Parse host/domain and port
    srv_mode = uri.startswith("mongodb+srv://")
    
    # Extract host info from the rightmost @
    prefix = "mongodb+srv://" if srv_mode else "mongodb://"
    rest = uri[len(prefix):]
    if "@" in rest:
        auth_part, host_part = rest.rsplit("@", 1)
    else:
        host_part = rest
    host_part = host_part.split("/")[0]
    
    if srv_mode:
        domain = host_part
        print(f"SRV Domain: {domain}")
    else:
        if ":" in host_part:
            domain, port_str = host_part.split(":")
            port = int(port_str)
        else:
            domain = host_part
            port = 27017
        print(f"Standard Host: {domain}:{port}")

    # 1. DNS Resolution
    print("\n[Step 1] DNS Resolution:")
    resolved_hosts = []
    if srv_mode:
        try:
            print(f"Querying SRV records for _mongodb._tcp.{domain}...")
            srv_records = dns.resolver.resolve(f"_mongodb._tcp.{domain}", 'SRV')
            for r in srv_records:
                target = str(r.target).rstrip('.')
                resolved_hosts.append((target, r.port))
                print(f"  Found SRV record: {target}:{r.port}")
        except Exception as e:
            print(f"  DNS RESOLUTION FAILED (SRV query error): {e}")
            print("MONGODB DNS RESOLUTION FAILED")
            return False
    else:
        try:
            ips = socket.gethostbyname_ex(domain)[2]
            for ip in ips:
                resolved_hosts.append((ip, port))
                print(f"  Resolved standard IP: {ip}")
        except Exception as e:
            print(f"  DNS RESOLUTION FAILED (Standard lookup error): {e}")
            print("MONGODB DNS RESOLUTION FAILED")
            return False

    if not resolved_hosts:
        print("  No hosts resolved.")
        print("MONGODB DNS RESOLUTION FAILED")
        return False
    
    print("  DNS Resolution: SUCCESS")

    # 2. TCP Connection
    print("\n[Step 2] TCP Connection:")
    tcp_success = False
    connected_host = None
    for host, p in resolved_hosts:
        try:
            print(f"  Attempting TCP connect to {host}:{p}...")
            # Try resolving host IP first
            ip = socket.gethostbyname(host)
            s = socket.create_connection((ip, p), timeout=5)
            s.close()
            print(f"  TCP Connection to {host}:{p} ({ip}): SUCCESS")
            tcp_success = True
            connected_host = (host, p, ip)
            break
        except Exception as e:
            print(f"  TCP Connection to {host}:{p} failed: {e}")
            
    if not tcp_success:
        print("  TCP CONNECTION FAILED: Could not establish TCP connection to any resolved host.")
        return False

    # 3. TLS/SSL Connection
    print("\n[Step 3] TLS/SSL Connection:")
    if srv_mode or "ssl=true" in uri.lower() or "tls=true" in uri.lower():
        try:
            host, p, ip = connected_host
            print(f"  Attempting TLS handshake with {host}:{p} ({ip})...")
            context = ssl.create_default_context()
            s = socket.create_connection((ip, p), timeout=5)
            ssl_conn = context.wrap_socket(s, server_hostname=host)
            ssl_conn.close()
            print("  TLS/SSL Connection: SUCCESS")
        except Exception as e:
            print(f"  TLS/SSL CONNECTION FAILED: {e}")
            return False
    else:
        print("  Skipping (non-SSL/TLS connection configured).")

    # 4. MongoDB Authentication & 5. Database access
    print("\n[Step 4 & 5] MongoDB Authentication and Database Access:")
    try:
        from backend.database.connection import clean_mongodb_uri
        escaped_uri = clean_mongodb_uri(uri)
        print("  Connecting with PyMongo Client...")
        client = MongoClient(escaped_uri, serverSelectionTimeoutMS=3000)
        # Attempt command to verify auth
        client.admin.command('ping')
        print("  MongoDB Authentication: SUCCESS")
        
        db = client[db_name]
        print(f"  Database Access to '{db_name}': SUCCESS")
    except OperationFailure as e:
        print(f"  MONGODB AUTHENTICATION FAILED: {e}")
        return False
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"  MONGODB SERVER CONNECTION FAILED: {e}")
        return False
    except Exception as e:
        print(f"  Unexpected error during client connection: {e}")
        return False

    # 6. Collection read/write (CRUD check)
    print("\n[Step 6] Collection Read/Write Check:")
    try:
        test_col = db["_connectivity_test"]
        print("  Inserting test document...")
        test_doc = {"test": "connectivity", "timestamp": socket.gethostname()}
        res = test_col.insert_one(test_doc)
        doc_id = res.inserted_id
        print(f"    Document inserted with ID: {doc_id}")
        
        print("  Querying test document...")
        found = test_col.find_one({"_id": doc_id})
        if found and found.get("test") == "connectivity":
            print("    Query verified successfully.")
        else:
            raise Exception("Document not found or mismatch.")
            
        print("  Deleting test document...")
        del_res = test_col.delete_one({"_id": doc_id})
        print(f"    Deleted {del_res.deleted_count} document(s).")
        print("  Collection read/write: SUCCESS")
    except Exception as e:
        print(f"  COLLECTION READ/WRITE FAILED: {e}")
        return False

    print("\n====================================================")
    print("ALL MONGODB DIAGNOSTICS PASSED SUCCESSFULLY")
    print("====================================================")
    return True

if __name__ == "__main__":
    success = run_diagnostics()
    sys.exit(0 if success else 1)
