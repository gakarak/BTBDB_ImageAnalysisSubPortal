'use strict';

angular.module('mainApp', ['ngMaterial']);

angular.module('mainApp')
.value('appConfig', {
    about: {
        version: '0.0.1',
        prjUrl: 'http://tuberculosis.by'
    }
}).controller('mainCtrl', ['$rootScope', '$scope', function($rootScope, $scope, $location) {
    console.log(':: MainApp->Contrller()');
}]);
