/**
 * Created by alexanderkalinovsky on 4/30/17.
 */
'use strict';

angular.module('dbStatus', ['ngMaterial','ngMessages'])
.component('dbStatus', {
    templateUrl: '/frontend/components/db-status/db-status.html',
    bindings: {
        dbstat: "<",
        dbstatStr: "<"
    },
    controller: function ($mdDialog, $http) {
        var self = this;

        self.$onInit = function () {
            self.updateInfo();
        };

        //////////////
        self.updateInfo = function() {
            $http({
                method: "GET",
                url: "/db/status/",
                data: {
                    procOnly: false
                }
            }).then(function (data) {
                // self.cases =
                try {
                    if(data.data.retcode === 0) {
                        self.dbstat = data.data.responce;
                        self.dbstatStr = JSON.stringify(self.dbstat,null,"    ");
                        console.log(self.dbstatStr);
                    } else {
                        console.log('Server return error: ' + data.data.errorstr);
                    }
                } catch (err) {
                    console.log('Error : [' + err + ']');
                }
                console.log(data);
            }, function (error) {
                console.log("error: show up error message" + error);
            });
        };
    }
});
