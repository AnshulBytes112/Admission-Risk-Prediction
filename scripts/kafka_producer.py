import argparse, csv, json, os, sys, time, uuid
from pathlib import Path
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

def wait_kafka(bootstrap, timeout=90):
    deadline=time.time()+timeout
    while time.time()<deadline:
        try:
            a=KafkaAdminClient(bootstrap_servers=bootstrap, request_timeout_ms=3000); a.close(); return
        except Exception as e:
            print(f"Waiting for Kafka: {e}", flush=True); time.sleep(3)
    raise TimeoutError(f"Kafka unavailable at {bootstrap}")

def ensure_topic(bootstrap, topic):
    a=KafkaAdminClient(bootstrap_servers=bootstrap, request_timeout_ms=5000)
    try:
        if topic not in a.list_topics():
            try: a.create_topics([NewTopic(topic, num_partitions=int(os.getenv("KAFKA_PARTITIONS", "3")), replication_factor=int(os.getenv("KAFKA_REPLICATION_FACTOR", "1")))])
            except TopicAlreadyExistsError: pass
    finally: a.close()

def value(v):
    if v=="": return None
    if v.lower()=="true": return True
    if v.lower()=="false": return False
    try: return float(v) if any(c in v for c in ".eE") else int(v)
    except ValueError: return v

def main():
    p=argparse.ArgumentParser(description="Publish synthetic observation rows to Kafka")
    p.add_argument("--input",default=os.getenv("INPUT_CSV","data/raw/patient_observations.csv"))
    p.add_argument("--topic",default=os.getenv("KAFKA_TOPIC","patient-observations"))
    p.add_argument("--bootstrap",default=os.getenv("KAFKA_BOOTSTRAP_SERVERS","localhost:9092"))
    p.add_argument("--delay-ms",type=int,default=int(os.getenv("KAFKA_PUBLISH_DELAY_MS","25")))
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--repeat",action="store_true")
    a=p.parse_args()
    path=Path(a.input)
    if not path.exists(): sys.exit(f"CSV not found: {path}. Generate synthetic data first.")
    if a.delay_ms<0: p.error("--delay-ms must be >= 0")
    wait_kafka(a.bootstrap); ensure_topic(a.bootstrap,a.topic)
    producer=KafkaProducer(bootstrap_servers=a.bootstrap,
        key_serializer=lambda k:k.encode() if isinstance(k,str) else k,
        value_serializer=lambda v:json.dumps(v,separators=(",",":")).encode(),
        acks="all", retries=10, linger_ms=10, max_in_flight_requests_per_connection=1)
    sent=0
    try:
        while True:
            with path.open(newline="",encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    event={k:value(v) for k,v in row.items()}
                    event["_stream"]={"event_key":str(event.get("event_id") or uuid.uuid4()),
                      "published_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                      "schema_version":"1.0","data_origin":"synthetic"}
                    producer.send(a.topic,key=str(event.get("patient_id","unknown")),value=event).get(timeout=30)
                    sent+=1
                    if sent%500==0: print(f"Published {sent} events",flush=True)
                    if a.delay_ms: time.sleep(a.delay_ms/1000)
                    if a.limit and sent>=a.limit: break
            producer.flush()
            if a.limit and sent>=a.limit: break
            if not a.repeat: break
        print(f"Published {sent} events to {a.topic}",flush=True)
    except KeyboardInterrupt: print(f"Stopped after {sent} events",flush=True)
    finally: producer.flush(); producer.close()
if __name__=="__main__": main()
