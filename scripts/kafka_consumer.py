import argparse, json, os, signal, time
from pathlib import Path
from kafka import KafkaConsumer

running=True
def stop(_sig,_frame):
    global running
    running=False

def wait_kafka(bootstrap, timeout=90):
    from kafka.admin import KafkaAdminClient
    deadline=time.time()+timeout
    while time.time()<deadline:
        try:
            a=KafkaAdminClient(bootstrap_servers=bootstrap,request_timeout_ms=3000); a.close(); return
        except Exception as e:
            print(f"Waiting for Kafka: {e}",flush=True); time.sleep(3)
    raise TimeoutError(f"Kafka unavailable at {bootstrap}")

def main():
    p=argparse.ArgumentParser(description="Consume Kafka observations to JSONL")
    p.add_argument("--topic",default=os.getenv("KAFKA_TOPIC","patient-observations"))
    p.add_argument("--bootstrap",default=os.getenv("KAFKA_BOOTSTRAP_SERVERS","localhost:9092"))
    p.add_argument("--output",default=os.getenv("CONSUMER_OUTPUT","data/streamed/patient_events.jsonl"))
    p.add_argument("--group",default=os.getenv("KAFKA_CONSUMER_GROUP","emergency-risk-consumer-v1"))
    p.add_argument("--max-messages",type=int,default=0)
    a=p.parse_args(); output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGINT,stop); signal.signal(signal.SIGTERM,stop)
    wait_kafka(a.bootstrap)
    consumer=KafkaConsumer(a.topic,bootstrap_servers=a.bootstrap,group_id=a.group,
      enable_auto_commit=False,auto_offset_reset="earliest",
      value_deserializer=lambda b:json.loads(b.decode()),
      key_deserializer=lambda b:b.decode() if b else None,
      max_poll_records=250)
    received=0
    print(f"Consuming {a.topic} → {output}",flush=True)
    try:
        with output.open("a",encoding="utf-8") as out:
            while running:
                records=consumer.poll(timeout_ms=1000,max_records=250)
                for partition,messages in records.items():
                    for m in messages:
                        out.write(json.dumps({"kafka_topic":m.topic,"kafka_partition":m.partition,
                          "kafka_offset":m.offset,"kafka_key":m.key,"consumed_at_utc":
                          time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"event":m.value},
                          separators=(",",":"))+"\n")
                        received+=1
                if records:
                    out.flush(); consumer.commit()
                if received and received%500==0: print(f"Consumed {received} messages",flush=True)
                if a.max_messages and received>=a.max_messages: break
        print(f"Stopped after {received} messages",flush=True)
    finally: consumer.close()
if __name__=="__main__": main()
