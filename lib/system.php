<?php
use Whoops\Handler\PrettyPageHandler;
use Whoops\Handler\JsonResponseHandler;
use Monolog\Logger;
use Monolog\Handler\StreamHandler;

$run     = new Whoops\Run;
$handler = new PrettyPageHandler;

$handler->addDataTable('Killer App Details', array(
  "Important Data" => 'Im super secret reporting data.',
  "Thingamajig-id" => 8675309
));

$handler->setPageTitle("We're all going to be fired!");
$run->pushHandler($handler);
$run->register();

/**
* Lets Try some different types of errors now:
* asdasd
* echo 'Look ma no semicolon'
* boom();		//undefined function name
* throw new Exception("Showing off advanced, better Exceptions with Whoops");
*/

/**
 *  Error Logging Setup
 *  First 4 lines are setup the add* lines actually start the logging.
 */
//$logger = new Impd\Logger();		//Creates a new custom logger in my namespace. ( How to load custom code )
$log = new Logger('AppLogger');
$log->pushHandler(new StreamHandler('logs/error.log', Logger::ERROR));
$log->pushHandler(new StreamHandler('logs/warning.log', Logger::WARNING));
$log->pushProcessor(function ($record) {			//Sets up extra info for every record
    $record['extra']['IP_Address']  = $_SERVER['SERVER_ADDR'];
    $record['extra']['SCRIPT_NAME'] = $_SERVER['SCRIPT_NAME'];
    return $record;
});

//Manual logging examples
//$log->addError('Bar', array('description', 'I am extra data that can be passed along with the record.'));
//$log->addWarning('Foo', array('message', 'boom that just happened.'));