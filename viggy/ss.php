// Database connection parameters
$servername = "your_mysql_server";
$username = "your_mysql_username";
$password = "your_mysql_password";
$dbname = "car_database";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Fetch cars from the database
$sql = "SELECT * FROM cars";
$result = $conn->query($sql);

// Process the result and send it as JSON
if ($result->num_rows > 0) {
    $cars = array();
    while ($row = $result->fetch_assoc()) {
        $cars[] = $row;
    }
    header('Content-Type: application/json');
    echo json_encode($cars);
} else {
    echo "0 results";
}

// Close the connection
