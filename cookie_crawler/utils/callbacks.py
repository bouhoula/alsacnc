from typing import Callable, Dict

from database.queries import update_entry


def get_callback(website: Dict) -> Callable[[bool], None]:
    def callback(success: bool) -> None:
        print(f"Finalizing for website {website['name']}")
        entry = {
            "success": int(success),
            "id": website["id"],
            "__tablename__": website["__tablename__"],
        }
        update_entry(entry)
        if success:
            print(f"Finished crawling website {website['name']}.")
        else:
            print(f"Crawling website {website['name']} failed.")

    return callback
