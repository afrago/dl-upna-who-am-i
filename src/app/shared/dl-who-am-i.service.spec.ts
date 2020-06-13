import { TestBed } from '@angular/core/testing';

import { DlWhoAmIService } from './dl-who-am-i.service';

describe('DlWhoAmIService', () => {
  let service: DlWhoAmIService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(DlWhoAmIService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
