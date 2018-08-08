import mysql.connector


class Database:
    def __init__(self, v):
        self.conn = None
        self.v = v

    def connect(self):
        try:
            self.conn = mysql.connector.connect(user='usergrid', password='jwflqs',
                host='10.101.58.155',
                database='entregas')
            print("Connected\n") if self.v >= 1 else None
            return self.conn
        except mysql.connector.Error as err:
          if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password") if self.v >= 1 else None
          elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist") if self.v >= 1 else None
          else:
            print(err) if self.v >= 1 else None
        else:
          self.conn.close()

    def close(self):
        self.conn.close()
