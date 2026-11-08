from database.db import (
    initialize_database,
    create_chat,
    save_message,
    get_chat_messages,
    get_recent_messages,
    chat_exists,
    rename_chat,
    delete_chat,
)

initialize_database()

chat = create_chat("Test Chat")
chat_id = chat["id"]

# chat_exists
assert chat_exists(chat_id) is True
assert chat_exists("fake_id") is False

# save 5 messages
for i in range(5):
    role = "user" if i % 2 == 0 else "assistant"
    save_message(chat_id, role, f"message {i}")

# get_recent_messages returns the last 3, oldest first
recent = get_recent_messages(chat_id, limit=3)
assert [m["content"] for m in recent] == ["message 2", "message 3", "message 4"]

# rename_chat
assert rename_chat(chat_id, "Renamed Chat") is True
assert rename_chat("fake_id", "Nope") is False

# saving to a missing chat raises a clear error
try:
    save_message("fake_id", "user", "hello")
    assert False, "should have raised ValueError"
except ValueError:
    pass

# delete_chat also deletes its messages
assert delete_chat(chat_id) is True
assert chat_exists(chat_id) is False
assert get_chat_messages(chat_id) == []

print("All checks passed")